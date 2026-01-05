from __future__ import annotations

from typing import Any, Dict, List
import math

import pandas as pd


class InsightAgent:
    """
    Generates explainable, actionable insights using ONLY computed values from df.

    Output contract:
      {
        "kpis": [{"title": str, "value": str, "context": str}],
        "summary": str,
        "distributions": [{"column": str, "top": [{"label": str, "count": int}]}],
        "trends": [{"time_field": str, "metric": str, "points": [{"date": str, "value": float}]}],
        "correlations": [{"a": str, "b": str, "corr": float}],
        "warnings": [str]
      }
    """

    def generate(self, *, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        if df is None or df.empty:
            return {
                "kpis": [{"title": "No Data", "value": "0 rows", "context": "Query returned no records"}],
                "summary": "No data returned for the query. Adjust filters or table selection.",
                "distributions": [],
                "trends": [],
                "correlations": [],
                "warnings": ["empty_result"],
            }

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        warnings: List[str] = []
        if len(df) > 2_000_000:
            warnings.append("large_dataframe_memory_risk")

        # KPI cards
        kpis = self._kpis_from_plan(df=df, plan=plan, numeric_cols=numeric_cols)
        if not kpis:
            kpis = [{"title": "Rows", "value": f"{len(df):,}", "context": "Returned rows"}]
            kpis.append({"title": "Columns", "value": f"{len(df.columns):,}", "context": "Returned columns"})
            if numeric_cols:
                col = numeric_cols[0]
                kpis.append({"title": f"Sum({col})", "value": self._fmt(df[col].sum()), "context": "Total"})
                kpis.append({"title": f"Avg({col})", "value": self._fmt(df[col].mean()), "context": "Mean"})

        # Distributions (top categories)
        distributions: List[Dict[str, Any]] = []
        for c in cat_cols[:3]:
            vc = df[c].astype(str).value_counts(dropna=False).head(10)
            distributions.append(
                {
                    "column": c,
                    "top": [{"label": str(idx), "count": int(val)} for idx, val in vc.items()],
                }
            )

        # Trend signals (if datetime + numeric)
        trends: List[Dict[str, Any]] = []
        if date_cols and numeric_cols:
            tcol = date_cols[0]
            ncol = numeric_cols[0]
            try:
                tmp = df[[tcol, ncol]].copy()
                tmp[tcol] = pd.to_datetime(tmp[tcol], errors="coerce")
                tmp = tmp.dropna(subset=[tcol])
                if not tmp.empty:
                    tmp["__date"] = tmp[tcol].dt.date
                    g = tmp.groupby("__date")[ncol].sum().sort_index()
                    if len(g) >= 3:
                        trends.append(
                            {
                                "time_field": tcol,
                                "metric": ncol,
                                "points": [{"date": str(k), "value": float(v)} for k, v in g.tail(60).items()],
                                "note": "Summed by date",
                            }
                        )
            except Exception:
                warnings.append("trend_calc_failed")

        # Correlation hints
        correlations: List[Dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr(numeric_only=True)
                pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        a, b = numeric_cols[i], numeric_cols[j]
                        val = corr.loc[a, b]
                        if pd.notna(val):
                            pairs.append((abs(float(val)), float(val), a, b))
                pairs.sort(reverse=True)
                for _, v, a, b in pairs[:5]:
                    correlations.append({"a": a, "b": b, "corr": v})
            except Exception:
                warnings.append("correlation_calc_failed")

        summary = f"Returned {len(df):,} rows and {len(df.columns):,} columns."
        if distributions:
            summary += f" Strongest categorical breakdown: {distributions[0]['column']}."
        if trends:
            summary += " Trend signals detected."
        if correlations:
            summary += f" Strongest numeric relationship: {correlations[0]['a']} vs {correlations[0]['b']} (corr={correlations[0]['corr']:.2f})."

        return {
            "kpis": kpis,
            "summary": summary,
            "distributions": distributions,
            "trends": trends,
            "correlations": correlations,
            "warnings": warnings,
        }

    def _kpis_from_plan(self, *, df: pd.DataFrame, plan: Dict[str, Any], numeric_cols: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        # Always include row count card
        out.append({"title": "Rows", "value": f"{len(df):,}", "context": "Returned rows"})

        metrics = plan.get("metrics", [])
        if not isinstance(metrics, list):
            return out

        for m in metrics:
            if not isinstance(m, dict):
                continue

            name = m.get("name")
            agg = (m.get("agg") or "").lower().strip()
            field = m.get("field")

            if not isinstance(name, str) or not name.strip():
                continue

            # Metric exists already as a numeric column (e.g., aggregated SQL output)
            if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
                v = df[name].dropna()
                if not v.empty:
                    out.append({"title": name, "value": self._fmt(float(v.iloc[0]) if len(v) == 1 else float(v.sum())), "context": "From query"})
                continue

            # Else compute from raw field if possible
            if isinstance(field, str) and field in df.columns and pd.api.types.is_numeric_dtype(df[field]):
                s = df[field].dropna()
                if s.empty:
                    continue

                if agg == "sum":
                    val = float(s.sum())
                    ctx = f"SUM({field})"
                elif agg in ("avg", "mean"):
                    val = float(s.mean())
                    ctx = f"AVG({field})"
                elif agg == "min":
                    val = float(s.min())
                    ctx = f"MIN({field})"
                elif agg == "max":
                    val = float(s.max())
                    ctx = f"MAX({field})"
                elif agg == "count":
                    val = float(len(s))
                    ctx = f"COUNT({field})"
                else:
                    val = float(s.sum())
                    ctx = f"SUM({field})"

                out.append({"title": name, "value": self._fmt(val), "context": ctx})

        return out

    def _fmt(self, x: Any) -> str:
        try:
            v = float(x)
        except Exception:
            return str(x)

        if math.isnan(v) or math.isinf(v):
            return "NA"
        if abs(v) >= 1_000_000_000:
            return f"{v / 1_000_000_000:.2f}B"
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"{v / 1_000:.2f}K"
        if abs(v) < 1:
            return f"{v:.4f}"
        return f"{v:,.2f}"