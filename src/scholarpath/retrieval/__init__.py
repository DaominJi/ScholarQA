"""Budgeted support-path retrieval (Section 4, Algorithm 2).

Main entry point: :class:`scholarpath.retrieval.pipeline.ScholarPathRetriever`.

Submodules (imported lazily so that light-weight modules such as
``retrieval.result`` and ``retrieval.link_prediction`` stay cheap to import):

* ``query_understanding`` -- Step 1, aspects A_q with weights and roles;
* ``grounding``           -- Step 2, scopes (u_q, m_q) and query-local anchors;
* ``objective``           -- Eq. (5)-(6), aspect instances, selection state;
* ``units`` / ``scoring`` -- direct paths, complementary path sets, Eq. (9), bounds;
* ``search``              -- Eq. (7), exact bound-guided search (Prop. 7), ranked join (Prop. 3);
* ``compose``             -- Step 6, hierarchical context and the composition call;
* ``link_prediction``     -- cross-taxonomy prior s_theta (Eq. 8, 12).
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "ScholarPathRetriever": "scholarpath.retrieval.pipeline",
    "analyze_query": "scholarpath.retrieval.query_understanding",
    "Aspect": "scholarpath.retrieval.query_understanding",
    "QueryAnalysis": "scholarpath.retrieval.query_understanding",
    "ground": "scholarpath.retrieval.grounding",
    "Grounding": "scholarpath.retrieval.grounding",
    "Anchor": "scholarpath.retrieval.grounding",
    "DirectPath": "scholarpath.retrieval.units",
    "ComplementaryPathSet": "scholarpath.retrieval.units",
    "SelectionState": "scholarpath.retrieval.objective",
    "AspectSpace": "scholarpath.retrieval.objective",
    "AdequacyScorer": "scholarpath.retrieval.scoring",
    "JunctionTable": "scholarpath.retrieval.scoring",
    "BoundGuidedSearch": "scholarpath.retrieval.search",
    "ExhaustiveGreedy": "scholarpath.retrieval.search",
    "build_search_space": "scholarpath.retrieval.search",
    "ranked_join": "scholarpath.retrieval.search",
    "assemble_context": "scholarpath.retrieval.compose",
    "compose_answer": "scholarpath.retrieval.compose",
    "ContextRenderer": "scholarpath.retrieval.compose",
    "NodeCostTable": "scholarpath.retrieval.compose",
    "make_search": "scholarpath.retrieval.search",
    "RetrievedUnit": "scholarpath.retrieval.result",
    "RetrievalResult": "scholarpath.retrieval.result",
    "LinkPredictor": "scholarpath.retrieval.link_prediction",
    "EmbeddingLinkPredictor": "scholarpath.retrieval.link_prediction",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'scholarpath.retrieval' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
