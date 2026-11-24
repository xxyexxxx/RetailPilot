"""Legacy policy protocol adapter; the adaptive state machine is shared."""
from .evidence_workflow import (  # noqa: F401
    FOLLOWUP_TOPICS,
    _bounded_call,
    _run_adaptive,
    evidence_size,
    select_evidence,
)


def run_policy_followup(**kwargs):
    from .evidence_runtime import policy_workflow
    return policy_workflow(**kwargs)
