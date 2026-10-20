"""All LLM prompts used by ScholarPath, its baselines, and its evaluation.

Each module defines prompt builders returning
:class:`scholarpath.llm.LLMRequest` objects and a ``MOCK_HANDLERS`` dict that
lets :class:`scholarpath.llm.MockLLM` answer those requests offline.

* :mod:`.taxonomy`   -- schema proposal, paper labelling, discovery, audits (Section 3)
* :mod:`.retrieval`  -- query understanding and answer composition (Section 4)
* :mod:`.baselines`  -- listwise reranking, RAPTOR summaries, LightRAG extraction, agents
* :mod:`.evaluation` -- LLM judges (rubric coverage, hypothesis judging, matching)
* :mod:`.benchmark`  -- held-out hypothesis query generation, application verification
"""
