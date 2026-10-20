"""Surrogate response model of quality-targeted acquisition (Section 3.2, App. B.1).

The two diagnostic rates of region g of node v respond to n additional
papers read by the LLM through logistic curves (Equation 13)

    m+_{vg,t}(n) = sigma(logit m^_{vg,t} - lambda^mis_{vg,t} n),
    e+_{vg,t}(n) = sigma(logit e^_{vg,t} - lambda^err_{vg,t} n),

and the predicted regional loss is l+_{vg,t}(n) = m+(n) + [1 - m+(n)] e+(n).
The slope vector lambda_{vg,t} = (lambda^mis, lambda^err) measures how
sensitive the region is to further reading; slopes may be negative because
schema revisions temporarily raise disagreements.

* :class:`RegionResponse` keeps a Gaussian posterior over lambda_{vg}. After a
  round in which n > 0 papers were read from region g, the observed decrease
  in log-odds divided by n is a noisy measurement of lambda and updates the
  posterior through a Gaussian likelihood (conjugate update). A Thompson
  sample [60] of the posterior gives the sampled curve l~_{vg,t}(n) that
  enters the allocation DP (Eq. 10).
* :class:`TransferredPrior` cold-starts new regions (Equation 14):

      p(lambda | h, H) = integral N(lambda; W h, Omega) p(W, Omega | H) dW dOmega,

  where h collects the region descriptors (depth, share, dispersion, child
  count, papers read, abstention rate a, disagreement rate d) and (W, Omega)
  are hierarchical parameters learned from the completed episodes H. We use
  Bayesian linear regression with a conjugate Gaussian prior on each row of
  W and an empirical-Bayes (shrunk residual) estimate of Omega; the
  predictive distribution is Gaussian with mean W^ h and covariance
  Omega^ + diag(h^T S_k h).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Rates are clipped to [RATE_FLOOR, 1 - RATE_FLOOR] before taking log-odds.
RATE_FLOOR = 1e-3


def sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: Any, floor: float = RATE_FLOOR) -> Any:
    q = np.clip(np.asarray(p, dtype=float), floor, 1.0 - floor)
    out = np.log(q) - np.log1p(-q)
    return float(out) if out.ndim == 0 else out


def logit_measurement_var(rate: float, count: float) -> float:
    """Delta-method variance of logit(r^) for a rate measured over ``count`` papers.

    Var[logit r^] ~= 1 / (N r (1 - r)); the rate is smoothed (Jeffreys) so the
    variance stays finite at 0 and 1.
    """
    n = max(float(count), 1.0)
    r = (float(rate) * n + 0.5) / (n + 1.0)
    return 1.0 / (n * r * (1.0 - r))


def response_curve(m_hat: float, e_hat: float, lam: Sequence[float], n_max: int) -> np.ndarray:
    """l+(n) = m+(n) + (1 - m+(n)) e+(n) for n = 0..n_max (Equation 13)."""
    n = np.arange(max(0, int(n_max)) + 1, dtype=float)
    m = sigmoid(logit(m_hat) - float(lam[0]) * n)
    e = sigmoid(logit(e_hat) - float(lam[1]) * n)
    return m + (1.0 - m) * e


# ---------------------------------------------------------------------------
# Region descriptors h_{vg} (Equation 14)
# ---------------------------------------------------------------------------

DESCRIPTOR_NAMES = (
    "depth",
    "share",
    "dispersion",
    "log_children",
    "log_papers_read",
    "abstention",
    "disagreement",
)


@dataclass
class RegionDescriptor:
    """Descriptor h_{vg,t} of a region (Appendix B.1)."""

    depth: float
    share: float
    dispersion: float
    child_count: float
    papers_read: float
    abstention: float
    disagreement: float

    def vector(self) -> np.ndarray:
        """Feature vector (counts on a log scale)."""
        return np.array(
            [
                float(self.depth),
                float(self.share),
                float(self.dispersion),
                math.log1p(max(0.0, float(self.child_count))),
                math.log1p(max(0.0, float(self.papers_read))),
                float(self.abstention),
                float(self.disagreement),
            ],
            dtype=float,
        )

    def to_list(self) -> List[float]:
        return [float(x) for x in self.vector()]

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def region_dispersion(vectors: np.ndarray) -> float:
    """1 - ||mean of unit vectors||: 0 for identical papers, ~1 for scattered ones."""
    if vectors.size == 0:
        return 0.0
    v = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.maximum(norms, 1e-12)
    return float(max(0.0, 1.0 - np.linalg.norm(v.mean(axis=0))))


# ---------------------------------------------------------------------------
# Per-region posterior (Equation 13 + Gaussian likelihood)
# ---------------------------------------------------------------------------


class RegionResponse:
    """Gaussian posterior N(mu, Sigma) over lambda = (lambda^mis, lambda^err) of one region."""

    def __init__(
        self,
        mean: Sequence[float],
        cov: np.ndarray,
        obs_noise_var: float = 0.02 ** 2,
        m_hat: float = 0.5,
        e_hat: float = 0.5,
    ):
        self.mean = np.asarray(mean, dtype=float).reshape(2)
        self.cov = _psd(np.asarray(cov, dtype=float).reshape(2, 2))
        self.obs_noise_var = float(obs_noise_var)
        self.m_hat = float(m_hat)
        self.e_hat = float(e_hat)
        self.observations: List[Tuple[int, float, float]] = []  # (n, slope_mis, slope_err)

    @classmethod
    def from_prior(cls, prior: Tuple[np.ndarray, np.ndarray], obs_noise_var: float) -> "RegionResponse":
        return cls(prior[0], prior[1], obs_noise_var)

    def set_rates(self, m_hat: float, e_hat: float) -> None:
        """Current proxies m^_{vg,t}, e^_{vg,t} (the curves start at l(0) = l^_{vg,t})."""
        self.m_hat = float(m_hat)
        self.e_hat = float(e_hat)

    @property
    def current_loss(self) -> float:
        return self.m_hat + (1.0 - self.m_hat) * self.e_hat

    def expected_curve(self, n_max: int) -> np.ndarray:
        """Curve at the posterior mean of lambda."""
        return response_curve(self.m_hat, self.e_hat, self.mean, n_max)

    def sample_slopes(self, rng: np.random.Generator) -> np.ndarray:
        L = np.linalg.cholesky(_psd(self.cov) + 1e-12 * np.eye(2))
        return self.mean + L @ rng.standard_normal(2)

    def thompson_curve(self, n_max: int, rng: np.random.Generator) -> np.ndarray:
        """Thompson sample [60]: draw lambda from the posterior, return l~(0..n_max)."""
        return response_curve(self.m_hat, self.e_hat, self.sample_slopes(rng), n_max)

    def observe(
        self,
        n: int,
        m_before: float,
        m_after: float,
        e_before: float,
        e_after: float,
        noise_var: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Conjugate update after a round that read ``n`` papers from this region.

        The observed slopes are the decreases in log-odds divided by n,
        s = ([logit m_before - logit m_after] / n, [logit e_before - logit e_after] / n),
        modelled as s ~ N(lambda, diag(noise_var)) (default: the configured
        ``obs_noise_var`` in both coordinates). Returns s, or None if n <= 0.
        """
        if n <= 0:
            return None
        s = np.array(
            [
                (logit(m_before) - logit(m_after)) / float(n),
                (logit(e_before) - logit(e_after)) / float(n),
            ],
            dtype=float,
        )
        if noise_var is None:
            nv = np.array([self.obs_noise_var, self.obs_noise_var], dtype=float)
        else:
            nv = np.maximum(np.asarray(noise_var, dtype=float).reshape(2), 1e-12)
        prec0 = np.linalg.inv(_psd(self.cov))
        prec_obs = np.diag(1.0 / nv)
        cov_post = np.linalg.inv(prec0 + prec_obs)
        self.mean = cov_post @ (prec0 @ self.mean + prec_obs @ s)
        self.cov = _psd(0.5 * (cov_post + cov_post.T))
        self.observations.append((int(n), float(s[0]), float(s[1])))
        return s

    def mean_observed_slope(self) -> Optional[np.ndarray]:
        """Paper-weighted mean of the observed slopes (None if never observed)."""
        if not self.observations:
            return None
        w = np.array([o[0] for o in self.observations], dtype=float)
        s = np.array([[o[1], o[2]] for o in self.observations], dtype=float)
        return (w[:, None] * s).sum(axis=0) / w.sum()


def _psd(cov: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Symmetrise and clip eigenvalues so the matrix is positive definite."""
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, floor)
    return (V * w) @ V.T


# ---------------------------------------------------------------------------
# Hierarchical transferred prior (Equation 14)
# ---------------------------------------------------------------------------


class TransferredPrior:
    """Cold-start prior over the slopes of a new region (Equation 14).

    The history H holds (h, lambda_obs) pairs of completed episodes: the
    descriptor of a region when its prior was set and the slope observed in
    that region during the episode. Given H,

    * each row w_k of W (k in {mis, err}) gets a conjugate Gaussian prior
      N(m0, S0) with intercept mean ``prior_mean`` and all variances
      ``prior_var`` (features are standardised with the history's moments);
    * Omega is estimated from the residuals with ``pseudo_obs`` pseudo
      observations shrinking it toward ``prior_var * I`` (empirical Bayes);
    * the posterior predictive for a new descriptor h is
      N(W^ phi(h), Omega^ + diag(phi^T S_k phi)).

    With an empty history, or when ``enabled`` is False (ablation
    "w/o transferred prior", Table 6), the prior is the fixed
    N(prior_mean * 1, prior_var * I).
    """

    def __init__(
        self,
        prior_mean: float = 0.05,
        prior_var: float = 0.05 ** 2,
        enabled: bool = True,
        pseudo_obs: float = 2.0,
        max_history: Optional[int] = 5000,
    ):
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)
        self.enabled = bool(enabled)
        self.pseudo_obs = float(pseudo_obs)
        self.max_history = max_history
        self.descriptors: List[List[float]] = []
        self.slopes: List[List[float]] = []
        self._fit: Optional[Dict[str, Any]] = None

    @classmethod
    def from_config(cls, config: Any) -> "TransferredPrior":
        """Build from a :class:`~scholarpath.config.BuilderConfig`."""
        return cls(config.prior_slope_mean, config.prior_slope_var, config.use_transferred_prior)

    # ----------------------------------------------------------- history
    @property
    def n_observations(self) -> int:
        return len(self.slopes)

    def fixed_prior(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.full(2, self.prior_mean), self.prior_var * np.eye(2)

    def update(self, descriptors: Sequence[Any], slopes: Sequence[Sequence[float]]) -> None:
        """Append (h, lambda_obs) pairs of a completed episode and refit (W, Omega)."""
        if len(descriptors) != len(slopes):
            raise ValueError("descriptors and slopes must be aligned")
        for h, s in zip(descriptors, slopes):
            vec = h.to_list() if isinstance(h, RegionDescriptor) else [float(x) for x in h]
            s = [float(x) for x in s]
            if len(s) != 2 or not all(np.isfinite(s)) or not all(np.isfinite(vec)):
                continue
            self.descriptors.append(vec)
            self.slopes.append(s)
        if self.max_history is not None and len(self.slopes) > self.max_history:
            self.descriptors = self.descriptors[-self.max_history :]
            self.slopes = self.slopes[-self.max_history :]
        self._fit = None

    # -------------------------------------------------------------- fit
    def _features(self, X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        Z = (X - mu) / sd
        return np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)

    def _refit(self) -> Dict[str, Any]:
        X = np.asarray(self.descriptors, dtype=float)
        Y = np.asarray(self.slopes, dtype=float)
        N, d = X.shape
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        Phi = self._features(X, mu, sd)
        p = Phi.shape[1]
        m0 = np.zeros(p)
        m0[0] = self.prior_mean
        S0_inv = np.eye(p) / self.prior_var
        omega = np.full(2, self.prior_var)
        means: List[np.ndarray] = [m0, m0]
        covs: List[np.ndarray] = [np.eye(p) * self.prior_var] * 2
        for _ in range(3):  # alternate posterior over W and the noise estimate
            means, covs = [], []
            for k in range(2):
                S = np.linalg.inv(S0_inv + Phi.T @ Phi / omega[k])
                m = S @ (S0_inv @ m0 + Phi.T @ Y[:, k] / omega[k])
                means.append(m)
                covs.append(0.5 * (S + S.T))
            resid = Y - np.stack([Phi @ means[0], Phi @ means[1]], axis=1)
            omega = (self.pseudo_obs * self.prior_var + (resid ** 2).sum(axis=0)) / (self.pseudo_obs + N)
        cross = float((resid[:, 0] * resid[:, 1]).sum()) / (self.pseudo_obs + N)
        lim = 0.95 * math.sqrt(omega[0] * omega[1])
        cross = max(-lim, min(lim, cross))
        Omega = np.array([[omega[0], cross], [cross, omega[1]]])
        return {"mu": mu, "sd": sd, "means": means, "covs": covs, "Omega": Omega}

    def predict(self, descriptor: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Prior (mean, covariance) of lambda for a region with descriptor h."""
        if not self.enabled or not self.slopes:
            return self.fixed_prior()
        if self._fit is None:
            self._fit = self._refit()
        f = self._fit
        h = descriptor.vector() if isinstance(descriptor, RegionDescriptor) else np.asarray(descriptor, dtype=float)
        phi = self._features(h.reshape(1, -1), f["mu"], f["sd"])[0]
        mean = np.array([phi @ f["means"][0], phi @ f["means"][1]])
        cov = f["Omega"].copy()
        cov[0, 0] += float(phi @ f["covs"][0] @ phi)
        cov[1, 1] += float(phi @ f["covs"][1] @ phi)
        return mean, _psd(cov)

    # ------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "enabled": self.enabled,
            "pseudo_obs": self.pseudo_obs,
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "descriptors": [list(map(float, h)) for h in self.descriptors],
            "slopes": [list(map(float, s)) for s in self.slopes],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TransferredPrior":
        prior = cls(
            float(d.get("prior_mean", 0.05)),
            float(d.get("prior_var", 0.05 ** 2)),
            bool(d.get("enabled", True)),
            float(d.get("pseudo_obs", 2.0)),
        )
        prior.update(d.get("descriptors", []), d.get("slopes", []))
        return prior
