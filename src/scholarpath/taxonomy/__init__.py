"""Taxonomy data structure, the node-wise confidence-bounded builder
(Algorithm 1, Section 3) and the selection-based baselines (Section 5.2)."""

from scholarpath.taxonomy.tree import CERTIFIED, LEAF, OPEN, UNRESOLVED, Taxonomy, TaxonomyNode

__all__ = [
    "CERTIFIED",
    "LEAF",
    "OPEN",
    "UNRESOLVED",
    "Taxonomy",
    "TaxonomyNode",
    # Algorithm 1 (lazily imported, see __getattr__)
    "AuditOutcome",
    "BatchPlan",
    "DeltaAllocator",
    "DynamicUpdater",
    "NodeWiseBuilder",
    "RegionResponse",
    "TransferredPrior",
    "UpdateReport",
    "audit_snapshot",
    "build_dual_taxonomies",
    "certificate_bound",
    "hypergeom_upper_bound",
    "plan_batch",
    "step_alpha",
]

_LAZY = {
    "AuditOutcome": "scholarpath.taxonomy.certification",
    "DeltaAllocator": "scholarpath.taxonomy.certification",
    "audit_snapshot": "scholarpath.taxonomy.certification",
    "certificate_bound": "scholarpath.taxonomy.certification",
    "hypergeom_upper_bound": "scholarpath.taxonomy.certification",
    "step_alpha": "scholarpath.taxonomy.certification",
    "BatchPlan": "scholarpath.taxonomy.allocation",
    "plan_batch": "scholarpath.taxonomy.allocation",
    "RegionResponse": "scholarpath.taxonomy.response_model",
    "TransferredPrior": "scholarpath.taxonomy.response_model",
    "NodeWiseBuilder": "scholarpath.taxonomy.builder",
    "build_dual_taxonomies": "scholarpath.taxonomy.builder",
    "DynamicUpdater": "scholarpath.taxonomy.dynamic",
    "UpdateReport": "scholarpath.taxonomy.dynamic",
}


def __getattr__(name: str):
    """Import the builder modules on first use (keeps ``import scholarpath.taxonomy`` light
    and free of import cycles with modules that only need the tree)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'scholarpath.taxonomy' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
