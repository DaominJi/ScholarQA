"""LLM-judge prompts used by the evaluation protocols (Section 5.3, Appendix C.3).

Every request built here is billed to the cost component ``"judge"``, which
:class:`scholarpath.llm.CostMeter` never counts as system cost, and is run at
temperature 0 ("judge models and prompts are fixed across all systems",
Appendix C.3).

Evidence benchmarks (official protocols, re-implemented in
:mod:`scholarpath.evaluation.qa_metrics`):

* ``evaluation.rubric_coverage`` -- ResearchQA rubric coverage ("Rub").
  The judge prompt is the one of ``build_prompt()`` in the official
  ``compute_coverage.py`` (github.com/realliyifei/ResearchQA, commit
  ``747a9a1``; also copied verbatim by allenai/olmo-eval and ported to the
  OpenAI v1 client by rlresearch/dr-tulu). Reproduced so that scores are
  comparable with the official leaderboard.
* ``evaluation.cs_property``, ``evaluation.cs_evidence``,
  ``evaluation.cs_claims`` -- ScholarQA-CS rubric correctness ("Corr"); the
  prompts of ``scripts/scholarqacs_rubric.py`` in
  github.com/AkariAsai/ScholarQABench (commit ``95e6fc5``, MIT License,
  (c) 2024 Akari Asai).
* ``evaluation.attribution`` -- statement-level citation support. The
  official ``scripts/citation_correctness_eval.py`` feeds the AttrScore
  prompt below to the NLI model ``osunlp/attrscore-flan-t5-xl``; the same
  prompt is used when an instruction-following LLM serves as the fallback
  attribution judge.
* ``evaluation.prometheus`` -- Prometheus-2 absolute grading of
  ScholarQA-Multi answers for organization / coverage / relevance; the prompt
  template of ``scripts/prometheus_eval.py`` and the three rubrics of
  ``rubrics/prometheus_rubrics_v8.json`` (ScholarQABench, MIT License).

Hypothesis queries (Section 5.3, Appendix C.3; prompts written for this
repository and fixed across systems):

* ``evaluation.hypothesis_match`` -- does a ranked list of proposed candidates
  realize the held-out counterpart of the expected answer record (Disc)?
* ``evaluation.hypothesis_score`` -- Feasibility / Novelty / Technical
  Soundness of one proposed idea and its rationale on a 0-100 scale.

Each task has a deterministic offline handler in :data:`MOCK_HANDLERS` that
answers from the request payload with lexical heuristics (no network).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from scholarpath.llm.base import LLMRequest, chat
from scholarpath.llm.mock import stable_hash
from scholarpath.text import tokenize

JUDGE_COMPONENT = "judge"

# ---------------------------------------------------------------------------
# ResearchQA rubric coverage (compute_coverage.py)
# ---------------------------------------------------------------------------

#: Ordinal labels of the ResearchQA judge, mapped to 1..5 by the official script.
COVERAGE_LABELS = ("Not at all", "Barely", "Moderately", "Mostly", "Completely")
COVERAGE_SCORES = {label: i + 1 for i, label in enumerate(COVERAGE_LABELS)}


def _coverage_prompt(response: str, questions: Sequence[str]) -> str:
    # Same text as ``build_prompt`` of the official ResearchQA compute_coverage.py.
    return (
        "Please judge the following questions based on the response below.\n"
        "For each question, select one of the following ratings to indicate the extent to which "
        "the response addresses the question:\n"
        "Not at all, Barely, Moderately, Mostly, Completely\n\n"
        "Definitions:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Only output one of the five phrases for each question, separated by newlines, and nothing else.\n\n"
        f"Response: {response}\n"
        "Questions:\n" + "\n".join(questions) + "\n\nOutput:"
    )


def rubric_coverage_request(
    response: str, questions: Sequence[str], model: Optional[str] = None
) -> LLMRequest:
    """One ResearchQA judge call for a batch of (at most 8) rubric items.

    The official script sends a single user message (no system prompt) at
    temperature 0.
    """
    return LLMRequest(
        task="evaluation.rubric_coverage",
        messages=[{"role": "user", "content": _coverage_prompt(response, questions)}],
        payload={"response": response, "questions": list(questions)},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        model=model,
    )


_LABEL_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\(?\d{1,2}[.):]|[A-Za-z]?\d{1,2}\s*[:.)-])\s*")


def normalize_coverage_label(line: str) -> Optional[str]:
    """Map one judge output line to a canonical coverage label (or ``None``).

    The official script requires each stripped line to equal a label exactly
    (anything else raises ``KeyError`` and aborts the run). We additionally
    tolerate list markers (``"1. "``, ``"- "``), surrounding quotes / markdown
    emphasis, trailing punctuation, letter case, and a ``"<question>: label"``
    echo; any other line is rejected.
    """
    text = line.strip()
    for _ in range(2):
        text = _LABEL_PREFIX_RE.sub("", text).strip()
    text = text.strip("*_`\"' ").rstrip(".;,!").strip("*_`\"' ")
    lowered = text.lower()
    for label in COVERAGE_LABELS:
        if lowered == label.lower():
            return label
    if ":" in text:
        return normalize_coverage_label(text.rsplit(":", 1)[1])
    return None


def parse_coverage_labels(text: str, expected: int) -> Optional[List[str]]:
    """Parse ``expected`` labels from a ResearchQA judge output.

    Mirrors the official check: split on newlines, drop empty lines, and
    require exactly one line per rubric item. Returns ``None`` for a malformed
    output (wrong number of lines or an unrecognized label).
    """
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    if len(lines) != expected:
        return None
    labels = [normalize_coverage_label(ln) for ln in lines]
    if any(lab is None for lab in labels):
        return None
    return [str(lab) for lab in labels]


# ---------------------------------------------------------------------------
# ScholarQA-CS rubric correctness (scholarqacs_rubric.py)
# ---------------------------------------------------------------------------

#: Fixed "expertise" criterion of RubricCorpusQaGenericMetric.score_output.
EXPERTISE_CRITERION = (
    "The level of expertise required to understand the answer should be roughly aligned with the "
    "estimated expertise of a typical person who would ask the question."
)

_CS_PROPERTY_SYSTEM = (
    "You will be given a question someone asked (in <question></question> tags) and the corresponding "
    "response (in <response></response> tags) given to them by an assistant.  You will then be given a "
    "specific criterion of the response to evaluate (in <criterion></criterion> tags).\n"
    "Return a score on a scale of 0 to 10 indicating how appropriate the response is based on the given "
    "criterion.  Judge only the specified aspect(s), not any other qualities of the answer.  Output JSON "
    # The official system prompt is a plain (non-f) string, so the doubled braces are sent literally.
    'in the format: {{"score": x}}.'
)


def cs_property_request(
    question: str, response: str, criterion: str, model: Optional[str] = None
) -> LLMRequest:
    """0-10 criterion score (``_score_property``; JSON mode, max 100 tokens)."""
    user = f"<question>{question}</question>\n<response>{response}</response>\n<criterion>{criterion}</criterion>"
    return LLMRequest(
        task="evaluation.cs_property",
        messages=chat(_CS_PROPERTY_SYSTEM, user),
        payload={"question": question, "response": response, "criterion": criterion},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        max_tokens=100,
        json_mode=True,
        model=model,
    )


def _cs_evidence_system(n: int) -> str:
    pad = " " * 14  # the official f-string keeps the source indentation of its continuation lines
    return (
        "You are given the response given by a scientific assistant to a user query enclosed in "
        "<response></response> tags.\n"
        f"{pad}In addition you are also given a list of snippets numbered from 1 to {n} enclosed in "
        "<snippets></snippets> tags, which should be present in the true answer. \n"
        f"{pad}Your job is to count how many snippets out of the given list are relevant to the provided "
        "response. \n"
        f"{pad}A snippet is relevant if the information provided by it is partly or completely present in the "
        "response. Count every snippet only once. \n"
        f'{pad}Output JSON with the count as a number in the format: {{"score": x}}.'
    )


def cs_evidence_request(response: str, evidence: Sequence[str], model: Optional[str] = None) -> LLMRequest:
    """Count of rubric evidence snippets present in the response (``_score_evidence``)."""
    snippets = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(evidence))
    user = f"<response>{response}</response>\n<snippets>{snippets}</snippets>"
    return LLMRequest(
        task="evaluation.cs_evidence",
        messages=chat(_cs_evidence_system(len(evidence)), user),
        payload={"response": response, "evidence": list(evidence)},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        max_tokens=100,
        json_mode=True,
        model=model,
    )


#: Output cap of the claims / citations / excerpts call. The official call sets
#: none (the provider maximum, 4096 for gpt-4-turbo); the claims JSON restates
#: the whole answer, so the client's default cap would truncate long answers.
CS_CLAIMS_MAX_TOKENS = 4096


def cs_claims_request(response: str, model: Optional[str] = None,
                      max_tokens: int = CS_CLAIMS_MAX_TOKENS) -> LLMRequest:
    """Split a response into claims / citations / excerpts (``_score_citations_excerpts_inner``)."""
    user = (
        "Here is a response to a question that includes several claims and citations:\n"
        f"Response: {response}\n\n"
        "Split the response into individual claims, citations, and excerpts from the citations, in JSON format: "
        '{"claims": [{"claim_text": "...", "citations": [{"citation_text": "...", "excerpts": ["...", ...]}, ...]}, ...]}'
        "\n\nIf a claim is missing citations or a citation is not accompanied by excerpts, some lists may be "
        "empty in your output."
    )
    return LLMRequest(
        task="evaluation.cs_claims",
        messages=chat("You are a helpful assistant.", user),
        payload={"response": response},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        max_tokens=int(max_tokens),
        json_mode=True,
        model=model,
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """JSON extraction with the semantics of ScholarQABench ``extract_json_from_response``.

    Takes the span from the first ``{`` to the last ``}``; if it does not
    parse, retries once with ``"]}"`` appended (truncated claim lists);
    returns ``None`` on failure (the caller then scores 0).
    """
    text = text or ""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        return None
    span = text[start:end] if end > start else text[start:]
    for candidate in (span, span + "]}"):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        return obj if isinstance(obj, dict) else None
    return None


# ---------------------------------------------------------------------------
# Statement-level citation attribution (citation_correctness_eval.py)
# ---------------------------------------------------------------------------

#: AttrScore prompt (``input_prompt`` of the official script); ``{claim}`` is the
#: citation-free statement and ``{output}`` the concatenated cited passages.
ATTRIBUTION_PROMPT = (
    "As an Attribution Validator, your task is to verify whether a given reference can support the given "
    "claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your "
    "response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A "
    "contradictory error occurs when you can infer that the answer contradicts the fact presented in the "
    "context, while an extrapolatory error means that you cannot infer the correctness of the answer based "
    "on the information provided in the context. \n\nClaim: {claim}\n Reference: {output}"
)
ATTRIBUTION_LABELS = ("Attributable", "Contradictory", "Extrapolatory")


def attribution_prompt(passage: str, claim: str) -> str:
    return ATTRIBUTION_PROMPT.format_map({"claim": claim, "output": passage})


def attribution_request(passage: str, claim: str, model: Optional[str] = None) -> LLMRequest:
    """LLM fallback for the AttrScore NLI judge (same prompt, one-word label)."""
    user = attribution_prompt(passage, claim) + "\n\nAnswer with exactly one word: Attributable, Contradictory or Extrapolatory."
    return LLMRequest(
        task="evaluation.attribution",
        messages=[{"role": "user", "content": user}],
        payload={"passage": passage, "claim": claim},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        max_tokens=10,
        model=model,
    )


_LABEL_RE = re.compile(r"\b(?:(not|non|un)[\s_-]*)?(attributable|contradictory|extrapolatory)\b")


def parse_attribution_label(text: str) -> Optional[str]:
    """Attribution label of an LLM answer (case-insensitive), else ``None``.

    The first word wins when it is a label (``"Attributable."``); otherwise
    the first label mentioned that is not negated -- ``"Not attributable"``,
    ``"non-attributable"`` and ``"unattributable"`` never read as
    ``Attributable`` (an unparsed answer counts as unsupported).
    """
    lowered = (text or "").strip().lower()
    first = re.match(r"[\W_]*([a-z]+)", lowered)
    if first and first.group(1) in {"attributable", "contradictory", "extrapolatory"}:
        return first.group(1).capitalize()
    for m in _LABEL_RE.finditer(lowered):
        if not m.group(1):
            return m.group(2).capitalize()
    return None


# ---------------------------------------------------------------------------
# Prometheus-2 absolute grading (prometheus_eval.py + prometheus_rubrics_v8.json)
# ---------------------------------------------------------------------------

PROMETHEUS_SYSTEM = "You are a fair evaluator language model."

#: Instruction prefix used for ScholarQA-Multi in the ScholarQABench README.
PROMETHEUS_INSTRUCTION = "Answer the question related to the most recent scientific literature."

#: The three rubrics of ``rubrics/prometheus_rubrics_v8.json`` (ScholarQABench, MIT License).
PROMETHEUS_RUBRICS: Dict[str, Dict[str, str]] = {
    "organization": {
        "criteria_description": (
            "Organization and Coherence: Evaluate if Output A is well-organized and logically structured. An "
            "acceptable response should: - be clearly structured, grouping related points together for a logical "
            "flow. - be coherent, without any contradictions or unnecessary repetition."
        ),
        "score1_description": (
            "Poor Organization; The response is disorganized, with no clear structure.  Points are scattered "
            "without any logical order, making it difficult to follow.  The text lacks coherence, and there are "
            "contradictions or irrelevant repetitions throughout. "
        ),
        "score2_description": (
            "Basic Organization; The response has some organization, but the structure is inconsistent.  There "
            "are occasional lapses in coherence, with minor contradictions or repetitive statements that disrupt "
            "the overall clarity. "
        ),
        "score3_description": (
            "Moderate Organization; The response is generally well-organized, with a clear structure that is "
            "mostly maintained; points are grouped logically, though there may be some minor lapses in flow or "
            "coherence. For instance, the answer has several clear paragraphs, each of which clearly discusses "
            "some core points. The text is mostly clear, with only occasional repetition or slight contradictions."
        ),
        "score4_description": (
            "Strong Organization; The response is well-organized, with a clear and logical structure that is "
            "followed consistently; Points are effectively grouped, and the flow is smooth. Several answers may "
            "come with clear opening or section / paragraph headers, which further improve the readability, "
            "although there are minor points (e.g., how to group papers together, section headers) that can be "
            "improved. There may be minor lapses in coherence, but overall, the response is clear and easy to "
            "follow and there is minimal repetition or contradictions."
        ),
        "score5_description": (
            "Exceptionally Organized; The response is exceptionally well-organized, with a flawless logical "
            "structure, and points are grouped perfectly, and the flow between them is seamless. The answer "
            "consists of multiple discussion points,  come with clear discourse markers or paragraphs / section "
            "headers. No need to fix the structure.  The text is coherent throughout, with no contradictions or "
            "unnecessary repetition."
        ),
    },
    "coverage": {
        "criteria_description": (
            "Coverage and Amount of Information: Evaluate if the output provides sufficient coverage and amount "
            "of information. The output should: - (coverage) Offer a comprehensive review of the area of "
            "interest, citing a diverse range of representative papers and discussing a variety of sources, not "
            "just a few (1-2 papers) \n- (depth) Provide enough relevant information to understand each "
            "discussion point.  "
        ),
        "score1_description": (
            "Severely Lacking Coverage; The output lacks coverage of several core lines of research or focuses "
            "predominantly on a single line of work, missing a holistic view of the area.\\n Greatly Limited "
            "Depth of Information; The output either lacks essential details needed to fully understand the "
            "topic (e.g., definitions of methods, relationships between methods).\\n"
        ),
        "score2_description": (
            "Partial Coverage; The output covers some key aspects of the area but misses significant lines of "
            "research or focuses too narrowly on a few sources. It lacks a well-rounded view and fails to "
            "adequately represent the diversity of work in the field.\\n Limited Amount of Information; The "
            "response provides some relevant information but leaves out important details that would help in "
            "understanding the topic fully. "
        ),
        "score3_description": (
            "Acceptable Coverage; The output discusses several representative works and provides a satisfactory "
            "overview of the area. However, including more papers or discussion points could enhance the answer "
            "significantly. It addresses the core aspects of the question but may miss some details.\\n "
            "Acceptable  Amount of Relevant Information; The output provides a reasonable amount of relevant "
            "information, though it may lack some helpful details.  "
        ),
        "score4_description": (
            "Good Coverage; The output offers good coverage of the area, discussing a variety of representative "
            "papers and sources. While it provides a broad overview, it may miss a few minor areas or additional "
            "papers that could enhance the comprehensiveness.\\n Mostly Sufficient Amount of Information; The "
            "response includes most of the necessary and relevant information to understand the topic. It avoids "
            "excessive irrelevant details, but a few points might benefit from deeper exploration or more "
            "specific examples."
        ),
        "score5_description": (
            "Comprehensive and Exceptional Coverage; The answer covers a diverse range of papers and viewpoints, "
            "offering a thorough overview of the area. It includes additional important discussion points not "
            "explicitly mentioned in the original question.\\n Necessary and Sufficient Amount of Information; "
            "The response provides all necessary and sufficient information, presenting it clearly without "
            "including irrelevant details that do not enhance the overall understanding."
        ),
    },
    "relevance": {
        "criteria_description": (
            "Relevance and Focus: Does the response stay on topic and maintain a clear focus to provide useful "
            "response to the question?"
        ),
        "score1_description": (
            "Off-topic; the content significantly deviates from the original question, making it difficult to "
            "discern its relevance or distract the user"
        ),
        "score2_description": (
            "Frequently Off-Topic with Limited Focus; The response addresses the question to some extent but "
            "often strays off-topic. There are several sections with irrelevant information or tangential points "
            "that do not contribute to answering the main question. These distractions make it difficult to "
            "maintain a clear focus and reduce the overall usefulness of the response. "
        ),
        "score3_description": (
            "Somewhat on topic but with several digressions or irrelevant information; while the response still "
            "centers around the original question, there are frequent deviations that distract from the main "
            "question or redundant information."
        ),
        "score4_description": (
            "Mostly On-Topic with Minor Deviations; The response stays largely on-topic, with a clear focus on "
            "addressing the question. However, there may be a few minor digressions or slightly irrelevant "
            "details that momentarily detract from the main focus. These deviations are infrequent and do not "
            "significantly undermine the overall clarity or usefulness of the response."
        ),
        "score5_description": (
            "Exceptionally focused and entirely on topic; the response remains tightly centered on the subject "
            "matter with enough depth and coverage of the core information, with every piece of information "
            "contributing directly to a comprehensive understanding of the topic."
        ),
    },
}


def _prometheus_prompt(instruction: str, response: str, reference_answer: Optional[str], rubric: Mapping[str, str]) -> str:
    head = (
        "An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets "
        "a score of 5, and a score rubric representing a evaluation criteria are given."
        if reference_answer is not None
        else "An instruction (might include an Input inside it), a response to evaluate and a score rubric "
        "representing a evaluation criteria are given."
    )
    steps = (
        "1. Write a detailed feedback that assess the quality of the response strictly based on the given score "
        "rubric, not evaluating in general."
        "2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the "
        "score rubric."
        '3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an '
        'integer number between 1 and 5)"'
        "4. Please do not generate any other opening, closing, and explanations."
    )
    scores = "".join(f"\nScore {k}: {rubric[f'score{k}_description']}" for k in range(1, 6))
    if reference_answer is not None:
        # ``prompt_template`` (with "###" headers) of prometheus_eval.py.
        body = (
            f"###The instruction to evaluate:\n{instruction}\n"
            f"###Response to evaluate:\n{response}\n"
            f"###Reference Answer (Score 5):\n{reference_answer}"
            "###Score Rubrics:\n"
        )
    else:
        # ``prompt_template_no_reference``.
        body = f"\n{instruction}\n{response}"
    return head + steps + body + f"\n\n[{rubric['criteria_description']}]" + scores


def prometheus_request(
    instruction: str,
    response: str,
    reference_answer: Optional[str],
    aspect: str,
    model: Optional[str] = None,
    rubric: Optional[Mapping[str, str]] = None,
    max_tokens: int = 512,
) -> LLMRequest:
    """Prometheus-2 absolute grading of one response for one aspect.

    ``instruction`` is the instruction prefix followed by the question (the
    official script concatenates them with a space). The official script
    wraps the prompt in fastchat's llama-2 template with the system message
    below; through an OpenAI-compatible endpoint the server applies the
    model's own chat template to the same (system, user) pair.
    """
    rub = dict(rubric or PROMETHEUS_RUBRICS[aspect])
    prompt = _prometheus_prompt(instruction, response, reference_answer, rub)
    return LLMRequest(
        task="evaluation.prometheus",
        messages=chat(PROMETHEUS_SYSTEM, prompt),
        payload={"response": response, "reference_answer": reference_answer, "aspect": aspect,
                 "instruction": instruction},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        max_tokens=max_tokens,
        model=model,
    )


def parse_prometheus_score(text: str) -> Optional[int]:
    """Integer score 1-5 from a Prometheus output (``[RESULT] k``), else ``None``.

    Follows ``get_grading_dict`` / ``extract_score`` of prometheus_eval.py:
    the text after ``"[RESULT] "`` up to the end of line, falling back to the
    phrase "the overall score is k" in the feedback. Whitespace and trailing
    punctuation after the digit are tolerated (the official parser only
    accepts the bare digit).
    """
    text = text or ""
    idx = text.find("[RESULT]")
    if idx >= 0:
        tail = text[idx + len("[RESULT]"):].lstrip(" :").split("\n")[0]
        m = re.match(r"\s*\(?([1-5])\)?(?:\s*(?:/\s*5)?\s*[.!]?\s*)$", tail)
        if m:
            return int(m.group(1))
    m = re.search(r"the overall score is (\d+)", text)
    if m and 1 <= int(m.group(1)) <= 5:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Held-out hypothesis queries: discovery matching and idea scoring
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = (
    "You are an expert scientist who evaluates research-ideation systems against ground truth. You are "
    "strict but fair about terminology: synonyms, acronyms and specific variants of the same concept count "
    "as the same concept, while merely related, broader, or neighbouring concepts do not. Answer with a "
    "single JSON object."
)


def hypothesis_match_request(
    query: str,
    counterpart: str,
    rationale: str,
    candidates: Sequence[Mapping[str, Any]],
    known_endpoint: Optional[str] = None,
    setting: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMRequest:
    """Does any proposed candidate realize the held-out counterpart? (Disc, App. C.1, C.3)

    ``candidates`` is the ranked list of proposed ideas
    (``{"name", "text", "rationale"}``); the judge returns the 1-based numbers
    of every candidate that realizes the expected answer record.
    """
    lines = []
    for i, c in enumerate(candidates, 1):
        rat = f" Rationale: {c['rationale']}" if c.get("rationale") else ""
        lines.append(f"{i}. {c.get('name') or '(unnamed)'}: {c.get('text', '')}{rat}")
    context = ""
    if known_endpoint:
        context += f"Known endpoint given in the question: {known_endpoint}\n"
    if setting:
        context += f"Setting: {setting.replace('_', ' ')}\n"
    user = (
        f"A research assistant answered the following research question by proposing unstudied directions.\n\n"
        f"Question: {query}\n{context}\n"
        "Expected answer record (the direction later realized by a published paper, hidden from the assistant):\n"
        f"- Held-out counterpart: {counterpart}\n"
        f"- Rationale: {rationale or '(none)'}\n\n"
        "Proposed candidates, ranked:\n" + ("\n".join(lines) or "(none)") + "\n\n"
        "A candidate realizes the expected answer if it proposes the held-out counterpart itself (the same "
        "concept, a synonym or acronym of it, or a specific variant that clearly is it) for this question. A "
        "broader family, a sibling technique, or a candidate that only mentions the counterpart in passing does "
        "not count.\n"
        'Output JSON: {"matches": [<numbers of the candidates that realize it>], "explanation": "<one sentence>"}'
    )
    return LLMRequest(
        task="evaluation.hypothesis_match",
        messages=chat(_MATCH_SYSTEM, user),
        payload={"query": query, "counterpart": counterpart, "rationale": rationale,
                 "candidates": [dict(c) for c in candidates]},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        max_tokens=300,
        json_mode=True,
        model=model,
    )


HYPOTHESIS_CRITERIA = ("feasibility", "novelty", "soundness")

_SCORE_SYSTEM = (
    "You are a senior reviewer for a top scientific venue who assesses proposed research ideas. You score "
    "each idea independently, on absolute 0-100 scales, and answer with a single JSON object."
)


def hypothesis_score_request(
    query: str,
    idea: str,
    rationale: Optional[str],
    evidence: Sequence[str] = (),
    judge_name: str = "",
    model: Optional[str] = None,
) -> LLMRequest:
    """Feasibility / Novelty / Technical Soundness of one proposed idea (Table 4).

    ``evidence`` optionally lists the titles of the papers the system cited
    for the idea. ``judge_name`` only enters the payload (offline mock).
    """
    ev = ""
    if evidence:
        ev = "Cited supporting papers:\n" + "\n".join(f"- {e}" for e in evidence) + "\n"
    user = (
        f"Research question: {query}\n\n"
        f"Proposed idea: {idea}\n"
        f"Rationale: {rationale or '(none given)'}\n{ev}\n"
        "Score the idea together with its rationale on each criterion from 0 (worst) to 100 (best):\n"
        "- Feasibility: can the idea be implemented and empirically validated with existing data, tools and "
        "compute within a typical research project?\n"
        "- Novelty: how original is the idea relative to the established literature on this question "
        "(0 = standard practice, 100 = a genuinely new direction)?\n"
        "- Technical Soundness: is the rationale technically correct, i.e., do the mechanism and assumptions "
        "of the proposal match what the problem requires, without logical gaps?\n"
        'Output JSON: {"feasibility": <0-100>, "novelty": <0-100>, "soundness": <0-100>, '
        '"explanation": "<two sentences at most>"}'
    )
    return LLMRequest(
        task="evaluation.hypothesis_score",
        messages=chat(_SCORE_SYSTEM, user),
        payload={"query": query, "idea": idea, "rationale": rationale or "", "evidence": list(evidence),
                 "judge": judge_name},
        component=JUDGE_COMPONENT,
        temperature=0.0,
        max_tokens=300,
        json_mode=True,
        model=model,
    )


def parse_hypothesis_scores(obj: Any) -> Optional[Dict[str, float]]:
    """``{"feasibility", "novelty", "soundness"}`` clipped to [0, 100], or ``None``."""
    if not isinstance(obj, Mapping):
        return None
    aliases = {
        "feasibility": ("feasibility", "feas", "feasible"),
        "novelty": ("novelty", "nov", "novel"),
        "soundness": ("soundness", "technical_soundness", "technical soundness", "sound"),
    }
    lowered = {str(k).strip().lower(): v for k, v in obj.items()}
    out: Dict[str, float] = {}
    for crit, keys in aliases.items():
        val = next((lowered[k] for k in keys if k in lowered), None)
        try:
            out[crit] = float(min(100.0, max(0.0, float(val))))
        except (TypeError, ValueError):
            return None
    return out


# ---------------------------------------------------------------------------
# Offline mock handlers (deterministic lexical heuristics)
# ---------------------------------------------------------------------------

_JUDGE_STOP = frozenset(
    """
    response answer does discuss discusses mention mentions explain explains describe describes cite cites
    study studies such title provide provides include includes address addresses detail details example
    examples near beginning briefly could should would answer's question specific
    """.split()
)


def _content(text: str) -> set:
    return {t for t in tokenize(text or "", remove_stopwords=True) if t not in _JUDGE_STOP}


def _recall(needle: str, haystack: str) -> float:
    """Fraction of the content tokens of ``needle`` present in ``haystack``."""
    need = _content(needle)
    if not need:
        return 0.0
    return len(need & _content(haystack)) / len(need)


def _mock_rubric_coverage(payload: Dict[str, Any], request: LLMRequest) -> str:
    response = payload.get("response", "")
    labels = []
    for q in payload.get("questions", []):
        r = _recall(q, response)
        idx = 4 if r >= 0.8 else 3 if r >= 0.6 else 2 if r >= 0.4 else 1 if r >= 0.2 else 0
        labels.append(COVERAGE_LABELS[idx])
    return "\n".join(labels)


def _mock_cs_property(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    if payload.get("criterion", "").startswith("The level of expertise"):
        n = len(tokenize(payload.get("response", "")))
        return {"score": 8 if n >= 40 else 5}
    return {"score": int(round(10 * _recall(payload.get("criterion", ""), payload.get("response", ""))))}


def _mock_cs_evidence(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    response = payload.get("response", "")
    return {"score": sum(1 for e in payload.get("evidence", []) if _recall(e, response) >= 0.5)}


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKER_RE = re.compile(r"\[([^\[\]]+)\]")
_QUOTE_RE = re.compile(r"\"([^\"]{8,})\"")


_REFERENCE_LINE = re.compile(r"^\[(\d+)\]\s*(.*\S)", re.MULTILINE)


def _mock_cs_claims(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    # A citation has an excerpt when the response quotes text or its References block
    # (convert_answer_scholar_cs.py format) lists a non-empty passage for it.
    body, _, refs = payload.get("response", "").partition("\nReferences:\n")
    passages = {m.group(1): m.group(2) for m in _REFERENCE_LINE.finditer(refs)}
    claims = []
    for sent in _SENT_SPLIT.split(body):
        sent = sent.strip(" -*\t")
        if len(sent) < 3:
            continue
        quotes = _QUOTE_RE.findall(sent)
        cites = [{"citation_text": f"[{m}]", "excerpts": quotes + ([passages[m][:200]] if m in passages else [])}
                 for m in _MARKER_RE.findall(sent)]
        claims.append({"claim_text": _MARKER_RE.sub("", sent).strip(), "citations": cites})
    return {"claims": claims}


def _mock_attribution(payload: Dict[str, Any], request: LLMRequest) -> str:
    return "Attributable" if _recall(payload.get("claim", ""), payload.get("passage", "")) >= 0.5 else "Extrapolatory"


def _mock_prometheus(payload: Dict[str, Any], request: LLMRequest) -> str:
    response = payload.get("response", "")
    reference = payload.get("reference_answer")
    if reference:
        r = _recall(reference, response)
    else:
        r = _recall(payload.get("instruction", ""), response)
    score = 1 + int(min(4, round(4 * min(1.0, 1.5 * r))))
    return (f"Feedback: The response reflects about {int(100 * r)}% of the reference content for the "
            f"{payload.get('aspect', 'given')} criterion. [RESULT] {score}")


def _mock_hypothesis_match(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    target = _content(payload.get("counterpart", ""))
    matches = []
    for i, cand in enumerate(payload.get("candidates", []), 1):
        name = _content(cand.get("name") or cand.get("text", ""))
        if target and name and (target <= name or len(target & name) / len(target | name) >= 0.5):
            matches.append(i)
    return {"matches": matches, "explanation": "Lexical match of candidate names (offline mock)."}


def _mock_hypothesis_score(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    idea = payload.get("idea", "")
    rationale = payload.get("rationale", "")
    grounding = _recall(idea, payload.get("query", "") + " " + rationale)
    jitter = stable_hash(payload.get("judge", ""), idea) % 11
    support = min(1.0, len(payload.get("evidence", [])) / 4.0)
    return {
        "feasibility": int(45 + 30 * support + jitter),
        "novelty": int(50 + 25 * (1.0 - grounding) + jitter),
        "soundness": int(35 + 30 * min(1.0, len(_content(rationale)) / 12.0) + 15 * support + jitter),
        "explanation": "Offline mock judgement.",
    }


MOCK_HANDLERS = {
    "evaluation.rubric_coverage": _mock_rubric_coverage,
    "evaluation.cs_property": _mock_cs_property,
    "evaluation.cs_evidence": _mock_cs_evidence,
    "evaluation.cs_claims": _mock_cs_claims,
    "evaluation.attribution": _mock_attribution,
    "evaluation.prometheus": _mock_prometheus,
    "evaluation.hypothesis_match": _mock_hypothesis_match,
    "evaluation.hypothesis_score": _mock_hypothesis_score,
}
