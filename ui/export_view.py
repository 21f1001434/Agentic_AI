from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st

from config import Settings
from traces.trace_store import TraceStore


def render_export(settings: Settings, trace_store: TraceStore) -> None:
    st.header("Export")

    runs = trace_store.list_runs()
    if not runs:
        st.info [st.info]("No runs found.")
        return

    run_ids = [r["run_id"] for r in runs]
    selected = st.selectbox("Select run_id", options=run_ids, index=0)

    run = trace_store.load_run(selected)
    if not run:
        st.error("Could not load run.")
        return

    status = run.get("status") or run.get("final", {}).get("status")
    st.markdown(f"**Status:** `{status}`")

    final = run.get("final", {})
    if status != "success":
        st.warning("Only successful runs can export dashboard/CSV.")
        st.download_button(
            "Download trace JSON (failed run)",
            data=json.dumps(run, indent=2),
            file_name=f"trace_{selected}.json",
            mime="application/json",
        )
        return

    dashboard_html = final.get("dashboard_html", "")
    if not dashboard_html:
        # maybe stored under nodes
        node = trace_store.get_node(selected, "J_dashboard__html")
        if isinstance(node, dict) and "html" in node:
            dashboard_html = node["html"]

    # CSV export: use df_preview if available
    df_preview = final.get("df_preview", [])
    df = pd.DataFrame(df_preview) if isinstance(df_preview, list) else pd.DataFrame()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if dashboard_html:
            st.download_button(
                "Download Dashboard HTML",
                data=dashboard_html,
                file_name=f"dashboard_{selected}.html",
                mime="text/html",
            )
        else:
            st.warning("No dashboard HTML found.")

    with col2:
        if not df.empty:
            st.download_button(
                "Download CSV (preview)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"data_preview_{selected}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No df_preview stored.")

    with col3:
        st.download_button(
            "Download trace JSON",
            data=json.dumps(run, indent=2),
            file_name=f"trace_{selected}.json",
            mime="application/json",
        )

    st.divider()
    st.subheader("Preview")
    if dashboard_html:
        st.caption("Dashboard HTML is downloadable above. Preview is in Ask Analytics.")
    if not df.empty:
        st.dataframe(df.head(50), use_container_width=True)