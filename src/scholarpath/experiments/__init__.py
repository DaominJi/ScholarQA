"""Reusable experiment drivers behind the ``scripts/run_*.py`` reproduction scripts.

The scripts under ``scripts/`` stay thin (argument parsing and output paths);
the logic that can be unit-tested lives in this package, for example

* :mod:`.common`   -- build caching keyed by (dataset, facet, method, seed,
  config hash), per-build cost accounting, corpus loading and subsampling,
  aggregation, and Markdown / CSV / plot writers shared by the experiments;
* :mod:`.taxonomy` -- the taxonomy-construction experiments of Section 5.2
  and Appendix C.4-C.5: Fig. 2 (soft node / edge F1 vs. selection ratio),
  Fig. 3 (relative LLM cost), Tables 8-9 (one-off vs. node-wise selection),
  Fig. 4 (scalability, Section 5.2.2), Table 6 (ablation) and the dynamic
  update study of Section 3.4 / Appendix B.3;
* :mod:`.qa`       -- the query-answering experiments of Section 5.3.

Submodules are imported explicitly (``from scholarpath.experiments import
taxonomy``); importing this package has no side effects.
"""

from __future__ import annotations

__all__ = ["common", "qa", "taxonomy"]
