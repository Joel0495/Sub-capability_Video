"""Model-based grader stubs.

These are fallback graders for low-verifiability tasks.
Prefer ruler graders whenever possible.
"""


def llm_as_judge_rubric(prediction: str, gt: str, rubric: str) -> float:
    """Use an LLM to judge answer quality against a rubric.

    TODO: Implement with actual LLM API call.
    Returns score in [0, 1].
    """
    raise NotImplementedError(
        "LLM-as-judge requires API configuration. "
        "Use ruler graders when possible."
    )


def evidence_consistency_judge(prediction: str, evidence: str) -> float:
    """Judge if the prediction is consistent with provided evidence.

    TODO: Implement with actual LLM API call.
    """
    raise NotImplementedError("Requires LLM API configuration.")


def forecast_semantic_match(prediction: str, gt: str) -> float:
    """Semantic matching for future event predictions.

    TODO: Implement with sentence similarity or LLM judge.
    """
    raise NotImplementedError("Requires embedding model or LLM API.")


def summary_rubric_judge(prediction: str, gt: str, rubric: str) -> float:
    """Judge summary quality against a rubric.

    TODO: Implement with actual LLM API call.
    """
    raise NotImplementedError("Requires LLM API configuration.")
