import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv()

from domain.data_understanding import DataUnderstandingArtifact
from domain.data_understanding_review import DataUnderstandingReview
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.data_loader import (
    UnsupportedFileTypeError,
    load_uploaded_dataset,
)
from tools.data_profiler import profile_dataframe
from tools.project_store import ProjectStore
from ui_data_preparation import (
    render_data_preparation_hitl_controls,
    render_data_preparation_review,
)
from ui_evaluation import (
    render_evaluation_hitl_controls,
    render_evaluation_review,
)
from ui_modeling import (
    render_modeling_hitl_controls,
    render_modeling_review,
)
from workflow.data_preparation_stage import (
    run_data_preparation_stage_for_project,
)
from workflow.data_understanding_stage import (
    run_data_understanding_stage_with_artifact,
)
from workflow.data_understanding_transitions import (
    evaluate_and_transition_data_understanding,
)
from ui_finalization import (
    render_finalization_hitl_controls,
    render_finalization_review,
)
from ui_finalization_revision import (
    render_finalization_revision_controls,
    render_finalization_revision_review,
)
from workflow.evaluation_stage import run_evaluation_stage
from workflow.finalization_stage import run_finalization_stage
from workflow.modeling_stage import run_modeling_stage_for_project
from workflow.states import WorkflowState

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

PROJECT_STORE = ProjectStore()
PERSISTENCE_DIR = Path("data/projects")
ACTIVE_STATE_PATH = PERSISTENCE_DIR / "active_state.json"
ACTIVE_DATASET_PREFIX = "active_dataset"


def _model_to_json(value: object) -> str | None:
    """Serialize a Pydantic model when one is present."""

    if value is None:
        return None

    return value.model_dump_json()


def _model_from_json(
    value: str | None,
    model_type: type,
) -> object | None:
    """Restore a Pydantic model from persisted JSON."""

    if not value:
        return None

    return model_type.model_validate_json(value)


def _persist_uploaded_dataset(uploaded_file: object) -> Path:
    """Persist the active dataset so it survives browser refreshes."""

    PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    dataset_path = PERSISTENCE_DIR / f"{ACTIVE_DATASET_PREFIX}{suffix}"

    for existing_path in PERSISTENCE_DIR.glob(f"{ACTIVE_DATASET_PREFIX}.*"):
        if existing_path != dataset_path and existing_path.exists():
            existing_path.unlink()

    dataset_path.write_bytes(uploaded_file.getvalue())

    st.session_state["persisted_dataset_path"] = str(dataset_path)
    st.session_state["persisted_dataset_name"] = uploaded_file.name

    return dataset_path


def _persist_active_state() -> None:
    """Persist the active project and UI state."""

    PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

    project = st.session_state.get("project_state")

    payload = {
        "project_id": (
            project.project_id
                if project is not None
                else st.session_state.get("project_id")
        ),
        "business_context": {
            key: st.session_state.get(key, "")
            for key in (
                "business_problem",
                "business_objective",
                "desired_outcome",
                "success_metrics",
                "constraints",
                "risks",
            )
        },
        "target_column": st.session_state.get("target_column"),
        "persisted_dataset_path": st.session_state.get(
            "persisted_dataset_path"
        ),
        "persisted_dataset_name": st.session_state.get(
            "persisted_dataset_name"
        ),
        "data_understanding_review": _model_to_json(
            st.session_state.get("data_understanding_review")
        ),
        "data_understanding_artifact": _model_to_json(
            st.session_state.get("data_understanding_artifact")
        ),
        "data_understanding_decision": _model_to_json(
            st.session_state.get("data_understanding_decision")
        ),
        "data_understanding_state": st.session_state.get(
            "data_understanding_state"
        ),
        "data_understanding_file_name": st.session_state.get(
            "data_understanding_file_name"
        ),
        "data_understanding_reviewer": st.session_state.get(
            "data_understanding_reviewer", ""
        ),
        "data_understanding_review_rationale": st.session_state.get(
            "data_understanding_review_rationale", ""
        ),
        "data_understanding_feedback": st.session_state.get(
            "data_understanding_feedback", ""
        ),
    }

    ACTIVE_STATE_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    if project is not None:
        PROJECT_STORE.save(project)


def _restore_persisted_state() -> None:
    """Restore the active workflow after a browser refresh."""

    if st.session_state.get("persistence_restored"):
        return

    st.session_state["persistence_restored"] = True

    if not ACTIVE_STATE_PATH.exists():
        return

    try:
        payload = json.loads(
            ACTIVE_STATE_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return

    for key, value in payload.get("business_context", {}).items():
        st.session_state[key] = value

    st.session_state["target_column"] = payload.get("target_column")
    st.session_state["persisted_dataset_path"] = payload.get(
        "persisted_dataset_path"
    )
    st.session_state["persisted_dataset_name"] = payload.get(
        "persisted_dataset_name"
    )

    for key in (
        "data_understanding_reviewer",
        "data_understanding_review_rationale",
        "data_understanding_feedback",
    ):
        st.session_state[key] = payload.get(key, "")

    st.session_state["data_understanding_file_name"] = payload.get(
        "data_understanding_file_name"
    )
    st.session_state["data_understanding_state"] = payload.get(
        "data_understanding_state"
    )

    review = _model_from_json(
        payload.get("data_understanding_review"),
        DataUnderstandingReview,
    )
    artifact = _model_from_json(
        payload.get("data_understanding_artifact"),
        DataUnderstandingArtifact,
    )
    decision = _model_from_json(
        payload.get("data_understanding_decision"),
        HITLDecision,
    )

    if review is not None:
        st.session_state["data_understanding_review"] = review
    if artifact is not None:
        st.session_state["data_understanding_artifact"] = artifact
    if decision is not None:
        st.session_state["data_understanding_decision"] = decision

    project_id = payload.get("project_id")
    if project_id:
        try:
            st.session_state["project_state"] = PROJECT_STORE.load(
                project_id
            )
        except FileNotFoundError:
            return
        except ValidationError:
            st.session_state.pop("project_state", None)
            st.session_state["persistence_restore_error"] = (
                "The previous project could not be restored because "
                "it was saved using an older project format. "
                "The saved session has been cleared. "
                "You can start a new project."
            )

            try:
                ACTIVE_STATE_PATH.unlink(missing_ok=True)
            except OSError:
                pass

            return


def _load_persisted_dataframe() -> pd.DataFrame | None:
    """Load the active persisted dataset, if one exists."""

    dataset_path = st.session_state.get("persisted_dataset_path")

    if not dataset_path:
        return None

    path = Path(dataset_path)
    if not path.exists():
        return None

    try:
        return load_uploaded_dataset(
            path.name,
            path.read_bytes(),
        )
    except (UnsupportedFileTypeError, ValueError, OSError):
        return None

def _load_prepared_dataframe(
    prepared_dataset_path: str | None,
) -> pd.DataFrame | None:
    """Load the prepared dataset produced by Data Preparation."""

    if not prepared_dataset_path:
        return None

    path = Path(prepared_dataset_path)

    if not path.exists():
        return None

    try:
        return load_uploaded_dataset(
            path.name,
            path.read_bytes(),
        )
    except (UnsupportedFileTypeError, ValueError, OSError):
        return None

def _start_over() -> None:
    """Delete persisted workflow state and reset the application."""

    project = st.session_state.get("project_state")

    if project is not None:
        PROJECT_STORE.delete(project.project_id)

    if ACTIVE_STATE_PATH.exists():
        ACTIVE_STATE_PATH.unlink()

    for dataset_path in PERSISTENCE_DIR.glob(f"{ACTIVE_DATASET_PREFIX}.*"):
        if dataset_path.exists():
            dataset_path.unlink()

    st.session_state.clear()
    st.rerun()



def render_global_start_over() -> None:
    """Render a global Start Over control for an active project."""

    if st.session_state.get("project_state") is None:
        return

    if not st.session_state.get("start_over_confirmation", False):
        if st.button(
            "Start Over",
            key="global_start_over",
        ):
            st.session_state["start_over_confirmation"] = True
            st.rerun()

        return

    st.warning(
        "Starting over will clear the current project, dataset, reviews, "
        "decisions, and workflow state. This cannot be undone."
    )

    cancel_column, confirm_column = st.columns(2)

    with cancel_column:
        if st.button(
            "Cancel",
            key="cancel_start_over",
        ):
            st.session_state["start_over_confirmation"] = False
            st.rerun()

    with confirm_column:
        if st.button(
            "Start Over",
            type="primary",
            key="confirm_start_over",
        ):
            _start_over()



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

    project_exists = st.session_state.get("project_state") is not None

    fields = {
        "business_problem": (
            "Business problem *",
            "Describe the business problem the data science project should address.",
        ),
        "business_objective": (
            "Business objective *",
            "What business objective should the analysis support?",
        ),
        "desired_outcome": (
            "Desired business outcome *",
            "What should improve or become possible if the project succeeds?",
        ),
        "success_metrics": (
            "Success metrics",
            "Describe measurable criteria that would indicate success.",
        ),
        "constraints": (
            "Known constraints",
            "Business, operational, technical, regulatory, or resource constraints.",
        ),
        "risks": (
            "Known risks",
            "Known risks, concerns, or assumptions that should be investigated.",
        ),
    }

    values: dict[str, object] = {}

    for key, (label, placeholder) in fields.items():
        st.session_state.setdefault(key, "")
        values[key] = st.text_area(
            label,
            placeholder=placeholder,
            key=key,
            disabled=project_exists,
        )

    return values


def render_dataset_upload() -> object:
    """Display the dataset uploader or the persisted active dataset."""

    st.subheader("Dataset")

    project = st.session_state.get("project_state")
    persisted_path = st.session_state.get("persisted_dataset_path")
    persisted_name = st.session_state.get("persisted_dataset_name")

    if project is not None and persisted_path:
        st.info(f"Active dataset: {persisted_name or Path(persisted_path).name}")
        return None

    uploaded_file = st.file_uploader(
        "Upload dataset *",
        type=["csv", "xlsx", "xls"],
        help="Upload the tabular dataset used by the data science workflow.",
    )

    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")
        _persist_uploaded_dataset(uploaded_file)
    elif persisted_path and Path(persisted_path).exists():
        st.info(
            f"Persisted dataset: {persisted_name or Path(persisted_path).name}"
        )

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

    persisted_target = st.session_state.get("target_column")
    if persisted_target not in options:
        persisted_target = "None"

    st.session_state.setdefault("target_column", persisted_target)

    selected_target = st.selectbox(
        "Target column (optional)",
        options,
        index=options.index(persisted_target),
        key="target_column",
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

    project = st.session_state.get("project_state")

    if project is not None and next_state == "next_stage":
        project.current_state = WorkflowState.DATA_PREPARATION
        st.session_state["project_state"] = project

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

    _restore_persisted_state()

    st.title("Multi-Agent Data Science AI")
    st.caption(
        "Turn a business problem and dataset into an "
        "evidence-based, human-reviewed ML experiment."
    )

    st.divider()

    render_global_start_over()

    left_column, right_column = st.columns(2)

    with left_column:
        business_context = render_business_context()

    with right_column:
        uploaded_file = render_dataset_upload()

    _persist_active_state()

    st.divider()

    render_workflow()

    if uploaded_file is not None:
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
    else:
        dataframe = _load_persisted_dataframe()

    if dataframe is None:
        st.warning(
            "DEBUG: persisted dataset could not be loaded."
        )
        return

    st.divider()

    render_dataset_preview(dataframe)
    target_column = render_target_selection(dataframe)
    
    _persist_active_state()

    st.divider()

    project = st.session_state.get("project_state")

    if project is None and st.button(
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

        dataset_path = st.session_state.get("persisted_dataset_path")
        dataset_name = st.session_state.get("persisted_dataset_name")

        if not dataset_path or not dataset_name:
            st.error("The uploaded dataset could not be persisted.")
            return

        project = ProjectState(
            project_id=f"project-{dataset_name}",
            project_name=business_context["business_problem"],
            dataset_path=dataset_path,
        )

        st.session_state["project_state"] = project
        st.session_state["project_id"] = project.project_id
        PROJECT_STORE.save(project)
        _persist_active_state()

        with st.spinner(
            "Preparing deterministic evidence and running the "
            "Data Understanding Agent..."
        ):
            try:
                artifact = profile_dataframe(
                    dataframe,
                    file_name=dataset_name,
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
        st.session_state["data_understanding_file_name"] = dataset_name
        _persist_active_state()

        st.rerun()

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
        _persist_active_state()

    project = st.session_state.get("project_state")

    st.write(
        "DEBUG:",
        project is not None,
        project.current_state if project is not None else None,
        project.data_preparation is None if project is not None else None,
    )

    if (
        project is not None
        and project.current_state == WorkflowState.DATA_PREPARATION
        and project.data_preparation is None
    ):
        with st.spinner(
            "Preparing deterministic evidence and running the "
            "Data Preparation Agent..."
        ):
            try:
                project = run_data_preparation_stage_for_project(
                    project,
                    dataframe,
                    st.session_state.get(
                        "persisted_dataset_name",
                        Path(project.dataset_path).name,
                    ),
                )
            except ValueError as exc:
                st.error(
                    f"Unable to complete Data Preparation: {exc}"
                )
                return
            except OSError as exc:
                st.error(
                    f"Unable to complete Data Preparation: {exc}"
                )
                return

        st.session_state["project_state"] = project
        PROJECT_STORE.save(project)
        _persist_active_state()

    if (
        project is not None
        and project.current_state
        == WorkflowState.AWAITING_PREPARATION_APPROVAL
        and project.data_preparation_review is not None
    ):
        st.divider()

        render_data_preparation_review(
            project.data_preparation_review,
        )

        st.divider()

        render_data_preparation_hitl_controls(
            project,
            dataframe,
        )
        _persist_active_state()

    project = st.session_state.get("project_state")

    if (
        project is not None
        and project.current_state == WorkflowState.MODELING
        and project.modeling is None
        and project.modeling_review is None
    ):
        modeling_target = (
            project.evaluation_history[-1].target_column
            if project.evaluation_history
            else st.session_state.get("target_column")
        )

        if not modeling_target:
            st.warning(
                "Select a target column before Modeling can begin. "
                "The target is required for deterministic Modeling evidence."
            )
            return

        dataset_name = st.session_state.get(
            "persisted_dataset_name",
            Path(project.dataset_path).name,
        )

        with st.spinner(
            "Preparing deterministic evidence and running the "
            "Modeling Agent..."
        ):
            try:
                prepared_dataframe = _load_prepared_dataframe(
                    project.prepared_dataset_path
                )

                if prepared_dataframe is None:
                    st.error(
                        "The prepared dataset could not be loaded for Modeling."
                    )
                    return

                project = run_modeling_stage_for_project(
                    project,
                    prepared_dataframe,
                    modeling_target,
                    Path(project.prepared_dataset_path).name,
                )

            except ValueError as exc:
                st.error(
                    f"Unable to complete Modeling: {exc}"
                )
                return

            except OSError as exc:
                st.error(
                    f"Unable to complete Modeling: {exc}"
                )
                return

        st.session_state["project_state"] = project
        PROJECT_STORE.save(project)
        _persist_active_state()
        st.rerun()

    project = st.session_state.get("project_state")

    if (
        project is not None
        and project.current_state == WorkflowState.EVALUATION
    ):
        with st.spinner(
            "Training the approved model and evaluating holdout results..."
        ):
            try:
                project = run_evaluation_stage(project)
            except (ValueError, OSError) as exc:
                st.error(f"Unable to complete Evaluation: {exc}")
                return

        st.session_state["project_state"] = project
        PROJECT_STORE.save(project)
        _persist_active_state()
        st.rerun()

    project = st.session_state.get("project_state")

    if (
        project is not None
        and project.modeling_review is not None
    ):
        st.divider()

        render_modeling_review(
            project.modeling_review,
        )

        if project.current_state in (
            WorkflowState.AWAITING_MODEL_SELECTION,
            WorkflowState.BLOCKED,
        ):
            st.divider()

            render_modeling_hitl_controls(
                project,
            )

        _persist_active_state()

    project = st.session_state.get("project_state")

    if (
        project is not None
        and project.current_state == WorkflowState.AWAITING_GO_NO_GO
    ):
        st.divider()

        if project.evaluation is None:
            st.error(
                "Evaluation evidence is missing. "
                "Human GO / NO-GO review cannot proceed."
            )
        else:
            render_evaluation_review(project.evaluation)

            st.divider()

            render_evaluation_hitl_controls(project)

    project = st.session_state.get("project_state")

    if (
        project is not None
        and project.current_state == WorkflowState.FINALIZATION
        and project.handoff_decision is not None
        and project.handoff_decision.decision == "request_revision"
    ):
        st.divider()
        render_finalization_revision_review(project)
        render_finalization_revision_controls(project)
        return

    if (
        project is not None
        and project.current_state == WorkflowState.FINALIZATION
    ):
        with st.spinner("Assembling verified POC handoff evidence..."):
            try:
                project = run_finalization_stage(project)
            except (ValueError, OSError) as exc:
                st.error(f"Unable to complete Finalization: {exc}")
                return

        st.session_state["project_state"] = project
        PROJECT_STORE.save(project)
        _persist_active_state()
        st.rerun()

    project = st.session_state.get("project_state")

    if (
        project is not None
        and project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    ):
        st.divider()

        if (
            project.finalization is None
            or not project.finalization_evidence_fingerprint
        ):
            st.error(
                "Verified Finalization evidence is missing. "
                "Human handoff review cannot proceed."
            )
        else:
            render_finalization_review(
                project.finalization,
                project.finalization_evidence_fingerprint,
            )

            st.divider()

            render_finalization_hitl_controls(project)

    project = st.session_state.get("project_state")

    if (
        project is not None
        and project.current_state in (
            WorkflowState.COMPLETE,
            WorkflowState.BLOCKED,
        )
        and project.handoff_decision is not None
    ):
        st.divider()
        st.subheader("Finalization Handoff Status")

        if (
            project.current_state == WorkflowState.COMPLETE
            and project.handoff_decision.decision == "approve"
        ):
            st.success("POC handoff approved and completed.")
        elif (
            project.current_state == WorkflowState.BLOCKED
            and project.handoff_decision.decision == "reject"
        ):
            st.error("POC handoff rejected. The workflow is blocked.")
        else:
            st.error("Handoff state and decision are inconsistent.")
            return

        st.write(
            f"**Reviewer:** {project.handoff_decision.reviewer}"
        )
        st.write(
            f"**Decision rationale:** "
            f"{project.handoff_decision.rationale}"
        )

        if project.finalization is not None:
            st.write(
                "**Handoff evidence SHA-256:** "
                f"`{project.finalization_evidence_fingerprint}`"
            )

        st.info(
            "This decision concerns the POC handoff only. "
            "Production deployment has not been authorized."
        )

if __name__ == "__main__":
    main()
