from domain.project_state import ProjectState
from tools.project_store import ProjectStore
from workflow.states import WorkflowState


def reload_data_preparation_project(
    project_id: str,
    *,
    store: ProjectStore | None = None,
) -> ProjectState:
    """Reload a persisted Data Preparation project without rerunning the agent."""

    project_store = store or ProjectStore()
    project = project_store.load(project_id)

    if project.current_state != WorkflowState.AWAITING_PREPARATION_APPROVAL:
        raise ValueError(
            "The persisted project is not awaiting Data Preparation approval."
        )

    if project.data_preparation is None:
        raise ValueError(
            "The persisted project has no Data Preparation artifact."
        )

    if project.data_preparation_review is None:
        raise ValueError(
            "The persisted project has no Data Preparation review."
        )

    return project
