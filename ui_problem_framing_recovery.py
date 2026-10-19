"""Review missing framing without discarding completed project stages."""

import streamlit as st

from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.problem_framing_recovery import reconstruct_problem_framing
from workflow.problem_framing_recovery import approve_recovered_framing


def render_problem_framing_recovery(project: ProjectState) -> ProjectState | None:
    st.subheader("Review Missing Problem Framing")
    st.warning(
        "This project did not save Problem Framing evidence. Review the saved business "
        "context below and approve it to continue. This records approval now and "
        "preserves your preparation, modeling, and evaluation results."
    )
    with st.form("recover_problem_framing"):
        context = {}
        for field, label in (
            ("business_problem", "Business problem *"),
            ("business_objective", "Business objective *"),
            ("desired_outcome", "Desired outcome *"),
            ("success_metrics", "Success criteria"),
            ("constraints", "Constraints"),
            ("risks", "Risks"),
        ):
            context[field] = st.text_area(
                label, value=st.session_state.get(field, ""), key=f"recovery_{field}"
            )
        reviewer = st.text_input("Reviewer *", key="framing_recovery_reviewer")
        rationale = st.text_area("Approval rationale *", key="framing_recovery_rationale")
        submitted = st.form_submit_button("Approve Framing and Continue", type="primary")

    if not submitted:
        return None
    try:
        artifact = reconstruct_problem_framing(context)
        decision = HITLDecision(
            decision="approve", reviewer=reviewer.strip(), rationale=rationale.strip()
        )
        return approve_recovered_framing(project, artifact, decision)
    except ValueError as exc:
        st.error(str(exc))
        return None
