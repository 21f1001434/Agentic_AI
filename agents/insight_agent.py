from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

import pandas as pd


class InsightAgent:
    """
    Generates explainable, actionable insights using ONLY computed values from df.

    Outputs include:
      - kpis: list of cards (title, value, delta optional, context)
      - summary: high-level overview
      - distributions: top categories
      - correlations: numeric correlation hints
      - warnings: data caveats
    """

    def generate(self, *, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        if df is None or df.empty:
            return {
                "kpis": [{"title": "No Data", "value": "0 rows", "context": "Query returned no records"}],
                "summary": "No data returned for the query. Adjust filters or table selection.",
                "warnings": ["empty_result"],
            }

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        warnings: List[str] = []
        if len(df) > 2_000_000:
            warnings.append("large_dataframe_memory_risk")

        # ---- KPI cards (from plan metrics when possible) ----
        kpis = self._kpis_from_plan(df, plan, numeric_cols)
        if not kpis:
            # fallback KPI set
            kpis = [
                {"title": "Rows", "value": f"{len(df):,}", "context": "Returned rows"},
                {"title": "Columns", "value": f"{len(df.columns):,}", "context": "Returned columns"},
            ]
            if numeric_cols:
                col = numeric_cols[0]
                kpis.append({"title": f"Avg {col}", "value": self._fmt(df[col].mean()), "context": "Mean value"})
                kpis.append({"title": f"Sum {col}", "value": self._fmt(df[col].sum()), "context": "Total value"})

        # ---- Top categories ----
        distributions: List[Dict[str, Any]] = []
        for c in cat_cols[:3]:
            vc = df[c].astype(str).value_counts(dropna=False).head(10)
            distributions.append(
                {
                    "column": c,
                    "top": [{"label": idx, "count": int(val)} for idx, val in vc.items()],
                }
            relate = " / ".join([d["column"] for d in distributions]))
        else:
            relate = "numeric focus" if numeric_cols else "unknown shape"

        # ---- Trend if datetime exists ----
        trends: List[Dict[str, Any]] = []
        if date_cols and numeric_cols:
            tcol = date_cols[0]
            ncol = numeric_cols[0]
            try:
                tmp = df[[tcol, ncol]].dropna()
                tmp[tcol] = pd.to_datetime(tmp[tcol], errors="coerce")
                tmp = tmp.dropna()
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

        # ---- Correlation hints ----
        correlations: List[Dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr(numeric_only=True)
                # pick top pairs
                pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        a, b = numeric_cols[i], numeric_cols[j]
                        val = corr.loc[a, b]
                        if pd.notna(val):
                            pairs.append((abs(val), float(val), a, b))
                pairs.sort(reverse=True)
                for absv, v, a, b in pairs[:5]:
                    correlations.append({"a": a, "b": b, "corr": v})
            except Exception:
                warnings.append("correlation_calc_failed")

        summary = (
            f"Returned {len(df):,} rows across {len(df.columns):,} columns. "
            f"Data appears centered around {relate}. "
            f"{'Trend signals detected.' if trends else ''}".strip()
        )

        return {
            "kpis": kpis,
            "summary": summary,
            "distributions": distributions,
            "trends": trends,
            "correlations": correlations,
            "warnings": warnings,
        }

    def _kpis_from_plan(self, df: pd.DataFrame, plan: Dict[str, Any], numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """
        If plan provides metrics like:
          {"name": "Total Revenue", "agg": "sum", "field": "amount"}
        compute them as KPI cards safely.
        """
        out: List[Dict[str, Any]] = []
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

            # If the SQLAgent already produced metric alias columns (aggregated),
            # that metric likely exists directly as a column in df.
            safe_alias = name.strip()
            if safe_alias in df.columns and pd.api.types.is_numeric_dtype(df[safe_alias]):
                val = float(df[safe_alias].dropna().sum()) if len(df) > 1 else float(df[safe_alias].iloc[0])
                out.append({"title": safe_alias, "value": self._fmt(val), "context": "From query result"})
                continue

            if isinstance(field, str) and field in df.columns and pd.api.types.is_numeric_dtype(df[field]):
                series = df[field].dropna()
                if series.empty:
                    continue
                if agg == "sum":
                    val = float(series.sum())
                elif agg in ("avg", "mean"):
                    val = float(series.mean())
                elif agg == "min":
                    val = float(series.min())
                elif agg == "max":
                    val = float(series.max())
                elif agg == "count":
                    val = float(len(series))
                else:
                    # default safe
                    val = float(series.sum())
                out.append({"title": safe_alias, "value": self._fmt(val), "context": f"{agg.upper()}({field})"})
        # Always include row count card
        out.insert(0, {"title": "Rows", "value": f"{len(df):,}", "context": "Returned rows"})
        return out

    def _fmt(self, x: Any) -> str:
        try:
            v = float(x)
        except Exception:
            return str(x)

        if math.isnan(v) or math.isinf(v):
            return "NA"
        if abs(v) >= 1_000_000_000:
            return f"{v/1_000_000_000:.2f}B"
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.2f}K"
        if abs(v) < 1:
            return f"{v:.4f}"
        return f"{v:,.2f}"