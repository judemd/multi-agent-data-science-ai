import pandas as pd
import streamlit as st

from domain.data_understanding import DataUnderstandingArtifact
from domain.data_understanding_review import DataUnderstandingReview
from domain.hitl_decision import HITLDecision
from tools.data_loader import (
    UnsupportedFileTypeError,
    load_uploaded_dataset,
)
from tools.data_profiler import profile_dataframe
from workflow.data_understanding_stage import (
    run_data_understanding_stage_with_artifact,
)
from workflow.data_understanding_transitions import (
    evaluate_and_transition_data_understanding,
)

st.set_page_config(
    page_title="Multi-Agent Data Science AI",
    layout="wide",
)


MASKED_COLUMN_TERMS = (
    "account",
    "customer_id",
    "email",
    "phone",
    "postcode",
    "postal",
    "address",
)


def render_workflow() -> None:
    """Display the current high-level workflow stages."""

    st.subheader("Workflow")

    stages = [
        "Problem Framing",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Final Recommendation",
    ]

    for index, stage in enumerate(stages):
        status = "Current" if index == 0 else "Pending"
        st.write(f"**{index + 1}. {stage}** - {status}")


def render_business_context() -> dict[str, object]:
    """Collect the business context required to begin analysis."""

    st.subheader("Business Context")

    business_problem = st.text_area(
        "Business problem *",
        placeholder=(
            "Describe the business problem the data science project "
            "should address."
        ),
    )

    business_objective = st.text_area(
        "Business objective *",
        placeholder=(
            "What business objective should the analysis support?"
        ),
    )

    desired_outcome = st.text_area(
        "Desired business outcome *",
        placeholder=(
            "What should improve or become possible if the project succeeds?"
        ),
    )

    success_metrics = st.text_area(
        "Success metrics",
        placeholder=(
            "Describe measurable criteria that would indicate success."
        ),
    )

    constraints = st.text_area(
        "Known constraints",
        placeholder=(
            "Business, operational, technical, regulatory, or resource constraints."
        ),
    )

    risks = st.text_area(
        "Known risks",
        placeholder=(
            "Known risks, concerns, or assumptions that should be investigated."
        ),
    )

    return {
        "business_problem": business_problem,
        "business_objective": business_objective,
        "desired_outcome": desired_outcome,
        "success_metrics": success_metrics,
        "constraints": constraints,
        "risks": risks,
    }


def render_dataset_upload() -> object:
    """Display the dataset uploader and return the uploaded file."""

    st.subheader("Dataset")

    uploaded_file = st.file_uploader(
        "Upload dataset *",
        type=["csv", "xlsx", "xls"],
        help="Upload the tabular dataset used by the data science workflow.",
    )

    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")

    return uploaded_file


def load_uploaded_dataframe(
    uploaded_file: object,
) -> pd.DataFrame:
    """Convert a Streamlit uploaded file into a pandas DataFrame."""

    return load_uploaded_dataset(
        uploaded_file.name,
        uploaded_file.getvalue(),
    )


def _should_mask_column(column_name: str) -> bool:
    """Determine whether a column should be masked in the browser preview."""

    normalized_name = column_name.lower()

    return any(
        term in normalized_name
        for term in MASKED_COLUMN_TERMS
    )


def _mask_value(value: object) -> str:
    """Return a privacy-conscious display representation of a value."""

    if pd.isna(value):
        return ""

    value_text = str(value)

    if len(value_text) <= 4:
        return "*" * len(value_text)

    return f"{value_text[:2]}{'*' * (len(value_text) - 4)}{value_text[-2:]}"


def build_safe_preview(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a display-only preview without changing analysis data."""

    preview = dataframe.head(10).copy()

    for column in preview.columns:
        if _should_mask_column(column):
            preview[column] = preview[column].map(_mask_value)

    return preview


def build_column_information(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build concise metadata for the browser column-information view."""

    information = pd.DataFrame(
        {
            "column": dataframe.columns,
            "datatype": [
                str(dtype)
                for dtype in dataframe.dtypes
            ],
            "missing_values": [
                int(dataframe[column].isna().sum())
                for column in dataframe.columns
            ],
            "unique_values": [
                int(dataframe[column].nunique(dropna=True))
                for column in dataframe.columns
            ],
        }
    )

    return information


def render_dataset_preview(
    dataframe: pd.DataFrame,
) -> None:
    """Display a structured, privacy-conscious dataset overview."""

    st.subheader("Dataset Preview")

    rows, columns = dataframe.shape

    first_metric, second_metric = st.columns(2)

    with first_metric:
        st.metric(
            "Rows",
            f"{rows:,}",
        )

    with second_metric:
        st.metric(
            "Columns",
            f"{columns:,}",
        )

    preview_tab, columns_tab = st.tabs(
        [
            "Preview rows",
            "Column information",
        ]
    )

    with preview_tab:
        st.caption(
            "Identifier-like fields are masked in this browser preview. "
            "The underlying dataset used for analysis is unchanged."
        )

        st.dataframe(
            build_safe_preview(dataframe),
            width="stretch",
            hide_index=True,
        )

    with columns_tab:
        st.dataframe(
            build_column_information(dataframe),
            width="stretch",
            hide_index=True,
        )

def render_target_selection(
    dataframe: pd.DataFrame,
) -> str | None:
    """Allow the user to explicitly select a binary numeric target."""

    candidates = [
        column
        for column in dataframe.select_dtypes(include="number").columns
        if set(dataframe[column].dropna().unique()).issubset({0, 1})
    ]

    options = ["None"] + candidates

    selected_target = st.selectbox(
        "Target column (optional)",
        options,
        help=(
            "Select a binary 0/1 target when target-aware EDA is required. "
            "The application does not infer the business target automatically."
        ),
    )

    if selected_target == "None":
        return None

    return selected_target

def render_data_understanding_evidence(
    artifact: DataUnderstandingArtifact,
    target_column: str | None = None,
) -> None:
    """Display the validated Data Understanding evidence."""


    st.subheader("Data Understanding")

    st.caption(
        "Deterministic evidence calculated from the uploaded dataset. "
        "These findings are evidence for review, not automated decisions."
    )

    st.write("### Dataset Health")

    first_column, second_column, third_column, fourth_column = st.columns(4)

    with first_column:
        st.metric(
            "Rows",
            f"{artifact.row_count:,}",
        )

    with second_column:
        st.metric(
            "Columns",
            f"{artifact.column_count:,}",
        )

    with third_column:
        st.metric(
            "Duplicate Rows",
            f"{artifact.duplicate_row_count:,}",
        )

    with fourth_column:
        missing_count = sum(
            artifact.missing_value_summary.values()
        )

        st.metric(
            "Missing Values",
            f"{missing_count:,}",
        )

    quality_tab, eda_tab, target_tab = st.tabs(
        [
            "Data Quality",
            "Exploratory Evidence",
            "Target Evidence",
        ]
    )

    with quality_tab:
        if artifact.potential_issues:
            st.write("#### Potential Issues")

            for issue in artifact.potential_issues:
                st.warning(issue)
        else:
            st.success(
                "No material data-quality issues were detected "
                "by the current deterministic checks."
            )

        if artifact.missing_value_summary:
            st.write("#### Missing / Null-like Values")

            missing_dataframe = pd.DataFrame(
                [
                    {
                        "column": column,
                        "missing_values": count,
                    }
                    for column, count in (
                        artifact.missing_value_summary.items()
                    )
                ]
            )

            st.dataframe(
                missing_dataframe,
                width="stretch",
                hide_index=True,
            )

        if artifact.outlier_summary:
            st.write("#### Potential Outliers")

            outlier_dataframe = pd.DataFrame.from_dict(
                artifact.outlier_summary,
                orient="index",
            )

            outlier_dataframe.index.name = "column"

            st.dataframe(
                outlier_dataframe,
                width="stretch",
            )

        if artifact.formatting_issues:
            st.write("#### Formatting Issues")
            st.json(artifact.formatting_issues)

    with eda_tab:
        if artifact.numeric_summary:
            st.write("#### Numeric Summary")

            numeric_dataframe = pd.DataFrame.from_dict(
                artifact.numeric_summary,
                orient="index",
            )

            numeric_dataframe.index.name = "column"

            st.dataframe(
                numeric_dataframe,
                width="stretch",
            )

        if artifact.numeric_correlations:
            st.write("#### Numeric Correlations")

            correlation_dataframe = pd.DataFrame(
                artifact.numeric_correlations
            )

            st.dataframe(
                correlation_dataframe,
                width="stretch",
            )

    with target_tab:
        if target_column is None:
            st.info(
                "No target column was selected. "
                "Target-specific evidence is not available."
            )
        else:
            st.write(
                f"#### Target: `{target_column}`"
            )

            if artifact.numeric_target_relationships:
                st.write(
                    "#### Numeric Relationships"
                )

                target_numeric_dataframe = (
                    pd.DataFrame.from_dict(
                        artifact.numeric_target_relationships,
                        orient="index",
                        columns=["correlation"],
                    )
                )

                target_numeric_dataframe.index.name = "feature"

                st.dataframe(
                    target_numeric_dataframe,
                    width="stretch",
                )

            if artifact.categorical_target_relationships:
                st.write(
                    "#### Categorical Target Rates"
                )

                categorical_rows = []

                for (
                    column,
                    values,
                ) in artifact.categorical_target_relationships.items():
                    for (
                        category,
                        statistics,
                    ) in values.items():
                        categorical_rows.append(
                            {
                                "feature": column,
                                "category": category,
                                "count": statistics["count"],
                                "target_rate": statistics["target_rate"],
                            }
                        )

                categorical_target_dataframe = pd.DataFrame(
                    categorical_rows
                )

                st.dataframe(
                    categorical_target_dataframe,
                    width="stretch",
                    hide_index=True,
                )

def render_agent_review(
    review: DataUnderstandingReview,
) -> None:
    """Display the Data Understanding Agent's structured review."""

    st.subheader("Data Understanding Agent Review")

    st.caption(
        "The agent interpreted deterministic evidence. "
        "Human review is required before workflow progression."
    )

    observed_tab, interpretation_tab, risks_tab, questions_tab, investigation_tab = (
        st.tabs(
            [
                "Observed Evidence",
                "Interpretation",
                "Risks & Limitations",
                "Human Review Questions",
                "Next Investigation",
            ]
        )
    )

    with observed_tab:
        if review.observed_evidence:
            for item in review.observed_evidence:
                st.write(f"- {item}")
        else:
            st.info("No observed evidence was returned.")

    with interpretation_tab:
        if review.interpretation:
            for item in review.interpretation:
                st.write(f"- {item}")
        else:
            st.info("No interpretation was returned.")

    with risks_tab:
        if review.risks_and_limitations:
            for item in review.risks_and_limitations:
                st.warning(item)
        else:
            st.success("No material risks or limitations were identified.")

    with questions_tab:
        if review.human_review_questions:
            for item in review.human_review_questions:
                st.write(f"- {item}")
        else:
            st.info("No human review questions were returned.")

    with investigation_tab:
        if review.recommended_next_investigation:
            for item in review.recommended_next_investigation:
                st.write(f"- {item}")
        else:
            st.info("No additional investigation was recommended.")

    if review.requires_human_review:
        st.warning(
            "The agent indicates that additional human review is required "
            "before progression."
        )
    else:
        st.info(
            "The agent does not indicate that additional review is required. "
            "Human approval is still required."
        )


def render_hitl_controls(
    review: DataUnderstandingReview,
) -> None:
    """Collect and evaluate the human Data Understanding decision."""

    st.subheader("Human Review")

    reviewer = st.text_input(
        "Reviewer *",
        placeholder="Enter reviewer name or role.",
        key="data_understanding_reviewer",
    )

    rationale = st.text_area(
        "Reviewer rationale *",
        placeholder="Explain the reason for your decision.",
        key="data_understanding_review_rationale",
    )

    feedback_text = st.text_area(
        "Requested changes / feedback",
        placeholder="Required when requesting a revision.",
        key="data_understanding_feedback",
    )

    approve_column, revision_column, reject_column = st.columns(3)

    with approve_column:
        approve = st.button(
            "Approve",
            type="primary",
            disabled=False,
        )

    with revision_column:
        request_revision = st.button("Request Revision")

    with reject_column:
        reject = st.button("Reject")

    decision_value = None

    if approve:
        decision_value = "approve"
    elif request_revision:
        decision_value = "request_revision"
    elif reject:
        decision_value = "reject"

    if decision_value is None:
        return

    if not reviewer.strip():
        st.warning("Enter the reviewer before submitting a decision.")
        return

    if not rationale.strip():
        st.warning("Enter a rationale before submitting a decision.")
        return

    feedback = []

    if decision_value == "request_revision":
        feedback = [
            item.strip()
            for item in feedback_text.splitlines()
            if item.strip()
        ]

        if not feedback:
            st.warning(
                "Provide at least one requested change before requesting a revision."
            )
            return

    decision = HITLDecision(
        decision=decision_value,
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        feedback=feedback,
    )

    next_state = evaluate_and_transition_data_understanding(
        review,
        decision,
    )

    st.session_state["data_understanding_decision"] = decision
    st.session_state["data_understanding_state"] = next_state

    if next_state == "next_stage":
        st.success(
            "Human approval recorded. Data Understanding may progress "
            "to the next stage."
        )
    elif next_state == "revision":
        st.warning(
            "Revision requested. Further Data Understanding work is required."
        )
    else:
        st.error(
            "The workflow is blocked by the current human review decision."
        )


def main() -> None:
    """Render the main application."""

    st.title("Multi-Agent Data Science AI")
    st.caption(
        "Turn a business problem and dataset into an "
        "evidence-based, human-reviewed ML experiment."
    )

    st.divider()

    left_column, right_column = st.columns(2)

    with left_column:
        business_context = render_business_context()

    with right_column:
        uploaded_file = render_dataset_upload()

    st.divider()

    render_workflow()

    if uploaded_file is None:
        return

    try:
        dataframe = load_uploaded_dataframe(
            uploaded_file,
        )
    except UnsupportedFileTypeError as exc:
        st.error(str(exc))
        return
    except (ValueError, OSError) as exc:
        st.error(f"Unable to load dataset: {exc}")
        return

    st.divider()

    render_dataset_preview(dataframe)
    target_column = render_target_selection(dataframe)

    st.divider()

    if st.button(
        "Analyze Dataset",
        type="primary",
    ):
        missing_fields = [
            field
            for field in (
                "business_problem",
                "business_objective",
                "desired_outcome",
            )
            if not business_context[field]
        ]

        if missing_fields:
            st.warning(
                "Please complete the required business context fields "
                "before analyzing the dataset."
            )
            return

        with st.spinner(
            "Preparing deterministic evidence and running the "
            "Data Understanding Agent..."
        ):
            try:
                artifact = profile_dataframe(
                    dataframe,
                    file_name=uploaded_file.name,
                )

                stage_result = run_data_understanding_stage_with_artifact(
                    artifact,
                    dataframe,
                    target_column=target_column,
                )

                artifact = stage_result.artifact
                review = stage_result.review

            except ValueError as exc:
                st.error(
                    f"The Data Understanding Agent returned an invalid review: {exc}"
                )
                return
            except OSError as exc:
                st.error(
                    f"Unable to complete Data Understanding: {exc}"
                )
                return

        st.session_state["data_understanding_review"] = review
        st.session_state["data_understanding_artifact"] = artifact
        st.session_state["data_understanding_file_name"] = uploaded_file.name

    artifact = st.session_state.get("data_understanding_artifact")
    review = st.session_state.get("data_understanding_review")

    if artifact is not None:
        st.divider()
        render_data_understanding_evidence(
            artifact,
            target_column=target_column,
        )

    if review is not None:
        st.divider()
        render_agent_review(review)

        st.divider()
        render_hitl_controls(review)


if __name__ == "__main__":
    main()

