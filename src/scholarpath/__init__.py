"""ScholarPath: multi-granularity semantic indexing and budgeted support-path
retrieval for evidence and hypothesis queries over scholarly repositories.

Main entry points:

* :class:`scholarpath.taxonomy.builder.NodeWiseBuilder` -- Algorithm 1
  (node-wise confidence-bounded taxonomy construction, Section 3);
* :class:`scholarpath.index.semantic_index.SemanticIndex` -- the
  multi-granularity semantic index (Section 2.1);
* :class:`scholarpath.retrieval.pipeline.ScholarPathRetriever` -- Algorithm 2
  (budgeted support-path retrieval, Section 4).
"""

from scholarpath.config import ScholarPathConfig
from scholarpath.types import Answer, AnswerItem, Chunk, Paper, Query

__version__ = "0.1.0"

__all__ = ["Answer", "AnswerItem", "Chunk", "Paper", "Query", "ScholarPathConfig", "__version__"]
