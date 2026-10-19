"""Independent reference exploration; never writes workflow evidence."""

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from tools.dtale_explorer import create_dtale_session, get_dtale_url


@st.cache_resource
def _explorer_executor() -> ThreadPoolExecutor:
    # D-Tale maintains a process-wide server, so serialize its initialization.
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="dtale-reference")


def _start_explorer(dataframe: pd.DataFrame) -> str:
    session = create_dtale_session(dataframe)
    if session is None:
        raise RuntimeError("D-Tale could not create an exploration session.")
    return get_dtale_url(session)


def render_dtale_reference(
    dataframe: pd.DataFrame,
    dataset_key: str,
    dataset_name: str,
) -> None:
    st.caption(
        "Reference exploration of the uploaded source dataset. D-Tale uses a separate "
        "copy; its edits and findings do not change workflow data, evidence, or approvals."
    )
    st.write(f"**Dataset:** {dataset_name}")
    st.write(f"**Rows:** {len(dataframe):,} · **Columns:** {len(dataframe.columns):,}")

    job = st.session_state.get("dtale_reference_job")
    if job is None or job["dataset_key"] != dataset_key:
        # Snapshot on the UI thread before handing data to the worker.
        future = _explorer_executor().submit(_start_explorer, dataframe.copy(deep=True))
        st.session_state["dtale_reference_job"] = {
            "dataset_key": dataset_key, "future": future,
        }
    _render_explorer_status()


@st.fragment(run_every="2s")
def _render_explorer_status() -> None:
    job = st.session_state.get("dtale_reference_job")
    if job is None:
        return
    future = job["future"]
    if not future.done():
        st.info("D-Tale is starting in the background. You can continue the workflow.")
        return
    try:
        url = future.result()
    except Exception as exc:  # noqa: BLE001 - optional explorer failures must stay local
        st.warning(f"D-Tale reference explorer is unavailable: {exc}")
        if st.button("Retry D-Tale", key="retry_dtale_reference"):
            st.session_state.pop("dtale_reference_job", None)
            st.rerun()
        return

    st.link_button("Open D-Tale Explorer", url)
    st.caption("Explore distributions, correlations, missing values, and charts in D-Tale.")
    if st.checkbox("Display D-Tale in this tab", key="embed_dtale_reference"):
        components.iframe(url, height=800, scrolling=True)
    if not os.getenv("DTALE_APP_ROOT"):
        st.caption("This local explorer is available from the computer running the app.")
