from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import uuid

import pandas as pd


class DashboardAgent:
    """
    Produces ONE self-contained HTML dashboard using Plotly.js.
    Features:
      - KPI cards from insights["kpis"]
      - Multi-chart grid auto layout
      - Uses plan["visuals"] if present; otherwise auto-detects charts from df
      - Includes a data table preview (first N rows)
    """

    def __init__(self, settings):
        self.settings = settings

    def build_dashboard(self, *, df: pd.DataFrame, plan: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        if df is None or df.empty:
            html = self._empty_dashboard("No data returned from query.")
            return {"html": html, "meta": {"status": "empty", "reason": "no rows"}}

        dashboard_id = f"dash_{uuid.uuid4().hex[:8]}"
        charts = self._build_chart_specs(df=df, plan=plan, insights=insights)

        # KPI cards
        kpis = insights.get("kpis", [])
        if not isinstance(kpis, list):
            kpis = []

        # Data preview
        preview_n = min(200, len(df))
        preview_rows = df.head(preview_n).to_dict(orient="records")
        columns = list(df.columns)

        html = self._render_html(
            dashboard_id=dashboard_id,
            kpis=kpis,
            charts=charts,
            columns=columns,
            preview_rows=preview_rows,
            summary=str(insights.get("summary", "")),
            warnings=insights.get("warnings", []),
        )

        meta = {
            "dashboard_type": "plotly_html",
            "charts": [{"title": c.get("title"), "type": c.get("type"), "x": c.get("x"), "y": c.get("y")} for c in charts],
            "kpis": kpis[:12],
            "rows": int(len(df)),
            "cols": len(df.columns),
        }
        return {"html": html, "meta": meta}

    # -----------------------------
    # Chart planning
    # -----------------------------
    def _build_chart_specs(self, *, df: pd.DataFrame, plan: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        # If plan defines visuals, honor them
        visuals = plan.get("visuals", [])
        if isinstance(visuals, list) and visuals:
            specs = []
            for v in visuals:
                if not isinstance(v, dict):
                    continue
                spec = self._normalize_visual(v, df)
                if spec:
                    specs.append(spec)
            if specs:
                return specs

        # Otherwise auto-detect charts
        return self._auto_charts(df)

    def _normalize_visual(self, v: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        vtype = (v.get("type") or "line").lower().strip()
        title = v.get("title") or "Chart"
        x = v.get("x")
        y = v.get("y")

        if isinstance(x, str) and x not in df.columns:
            return None
        if isinstance(y, str) and y not in df.columns:
            return None
        if isinstance(y, list):
            y = [c for c in y if c in df.columns]
            if not y:
                return None

        if vtype not in {"line", "bar", "scatter", "area", "hist"}:
            vtype = "line"

        return {
            "id": f"chart_{uuid.uuid4().hex[:8]}",
            "type": vtype,
            "title": str(title),
            "x": x,
            "y": y,
        }

    def _auto_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        charts: List[Dict[str, Any]] = []
        # 1) Trend: date vs first numeric
        if date_cols and numeric_cols:
            charts.append({
                "id": f"chart_{uuid.uuid4().hex[:8]}",
                "type": "line",
                "title": f"Trend: {numeric_cols[0]} over time",
                "x": date_cols[0],
                "y": numeric_cols[0],
            })

        # 2) Top categories (bar)
        if cat_cols and numeric_cols:
            charts.append({
                "id": f"chart_{uuid.uuid4().hex[:8]}",
                "type": "bar",
                "title": f"Top categories: {numeric_cols[0]} by {cat_cols[0]}",
                "x": cat_cols[0],
                "y": numeric_cols[0],
                "aggregate": "sum",
                "top_n": 20,
            })

        # 3) Scatter numeric-numeric
        if len(numeric_cols) >= 2:
            charts.append({
                "id": f"chart_{uuid.uuid4().hex[:8]}",
                "type": "scatter",
                "title": f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
            })

        # 4) Histogram
        if numeric_cols:
            charts.append({
                "id": f"chart_{uuid.uuid4().hex[:8]}",
                "type": "hist",
                "title": f"Distribution: {numeric_cols[0]}",
                "x": numeric_cols[0],
                "y": None,
            })

        # If still nothing, fallback: first 2 cols line
        if not charts and len(df.columns) >= 2:
            charts.append({
                "id": f"chart_{uuid.uuid4().hex[:8]}",
                "type": "line",
                "title": "Auto Chart",
                "x": df.columns[0],
                "y": df.columns[1],
            })

        return charts

    # -----------------------------
    # HTML rendering
    # -----------------------------
    def _render_html(
        self,
        *,
        dashboard_id: str,
        kpis: List[Dict[str, Any]],
        charts: List[Dict[str, Any]],
        columns: List[str],
        preview_rows: List[Dict[str, Any]],
        summary: str,
        warnings: Any,
    ) -> str:
        # Prepare chart container divs
        chart_divs = "\n".join(
            [f'<div class="chart-card"><div class="chart-title">{self._esc(c["title"])}</div><div id="{c["id"]}" class="chart"></div></div>' for c in charts]
        )

        # KPI cards
        kpi_divs = "\n".join(
            [
                f"""
                <div class="kpi">
                    <div class="kpi-title">{self._esc(str(k.get("title","")))}</div>
                    <div class="kpi-value">{self._esc(str(k.get("value","")))}</div>
                    <div class="kpi-context">{self._esc(str(k.get("context","")))}</div>
                </div>
                """
                for k in kpis[:12]
            ]
        )

        return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Agentic Analytics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js [cdn.plot.ly]"></script>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 16px;
  background: #f6f7fb;
}}
.header {{
  display:flex; flex-direction:column; gap:8px;
  margin-bottom: 12px;
}}
.h1 {{
  font-size: 20px;
  font-weight: 700;
}}
.sub {{
  color:#555;
}}
.kpi-grid {{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}}
.kpi {{
  background: white;
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}}
.kpi-title {{ font-size: 12px; color:#666; }}
.kpi-value {{ font-size: 22px; font-weight: 800; margin-top: 6px; }}
.kpi-context {{ font-size: 12px; color:#777; margin-top: 6px; }}
.grid {{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 14px;
}}
.chart-card {{
  background: white;
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}}
.chart-title {{
  font-weight: 700;
  margin-bottom: 8px;
}}
.chart {{
  height: 320px;
}}
.table-wrap {{
  background:white;
  margin-top: 16px;
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.07);
  overflow-x:auto;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}}
th, td {{
  border-bottom: 1px solid #eee;
  padding: 8px;
  text-align: left;
  white-space: nowrap;
}}
th {{
  background: #fafafa;
}}
.badge {{
  display:inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  background: #fff3cd;
  border: 1px solid #ffeeba;
  color:#856404;
  font-size: 12px;
}}
</style>
</head>

<body>
  <div class="header">
    <div class="h1">📊 Agentic Analytics Dashboard</div>
    <div class="sub">{self._esc(summary)}</div>
    <div>{self._render_warnings(warnings)}</div>
  </div>

  <div class="kpi-grid">
    {kpi_divs}
  </div>

  <div class="grid">
    {chart_divs}
  </div>

  <div class="table-wrap">
    <div style="font-weight:700;margin-bottom:8px;">Data Preview (first {len(preview_rows)} rows)</div>
    {self._render_table(columns, preview_rows)}
  </div>

<script>
const DATA = {json.dumps(preview_rows)};
const FULL_COLUMNS = {json.dumps(columns)};
const CHARTS = {json.dumps(charts)};

function isNumber(x) {{
  return typeof x === 'number' && !isNaN(x);
}}

function aggByCategory(rows, xcol, ycol, topN) {{
  const map = new Map();
  for (const r of rows) {{
    const k = (r[xcol] === null || r[xcol] === undefined) ? "NULL" : String(r[xcol]);
    const v = Number(r[ycol]);
    if (!isNaN(v)) {{
      map.set(k, (map.get(k) || 0) + v);
    }}
  }}
  const arr = Array.from(map.entries()).map(([k,v]) => ({{k, v}}));
  arr.sort((a,b) => b.v - a.v);
  const top = arr.slice(0, topN || 20);
  return {{
    x: top.map(o => o.k),
    y: top.map(o => o.v)
  }};
}}

function makeChart(spec) {{
  const id = spec.id [spec.id];
  const type = spec.type || 'line';
  const x = spec.x;
  const y = spec.y;

  if (type === 'hist') {{
    const vals = DATA.map(r => Number(r[x])).filter(v => !isNaN(v));
    Plotly.newPlot(id, [{{x: vals, type:'histogram'}}], {{
      margin: {{t: 10, l: 40, r: 10, b: 40}},
    }});
    return;
  }}

  if (!x || !y) {{
    Plotly.newPlot(id, [], {{title:'No valid x/y', margin: {{t: 20}}}});
    return;
  }}

  // bar with aggregation
  if (type === 'bar' && spec.aggregate) {{
    const agg = aggByCategory(DATA, x, y, spec.top_n || 20);
    Plotly.newPlot(id, [{{x: agg.x, y: agg.y, type:'bar'}}], {{
      margin: {{t: 10, l: 40, r: 10, b: 80}},
      xaxis: {{tickangle: -30}},
    }});
    return;
  }}

  // scatter
  if (type === 'scatter') {{
    const xs = DATA.map(r => r[x]);
    const ys = DATA.map(r => r[y]);
    Plotly.newPlot(id, [{{x: xs, y: ys, mode:'markers', type:'scatter'}}], {{
      margin: {{t: 10, l: 40, r: 10, b: 40}},
    }});
    return;
  }}

  // default line/area
  const xs = DATA.map(r => r[x]);
  const ys = DATA.map(r => r[y]);
  const trace = {{
    x: xs,
    y: ys,
    mode: 'lines+markers',
    type: 'scatter',
    fill: (type === 'area') ? 'tozeroy' : 'none',
  }};
  Plotly.newPlot(id, [trace], {{
    margin: {{t: 10, l: 40, r: 10, b: 40}},
  }});
}}

for (const spec of CHARTS) {{
  try {{
    makeChart(spec);
  }} catch (e) {{
    console.log("chart error", spec, e);
  }}
}}
</script>
</body>
</html>
"""

    def _render_table(self, cols: List[str], rows: List[Dict[str, Any]]) -> str:
        head = "".join([f"<th>{self._esc(c)}</th>" for c in cols])
        body_rows = []
        for r in rows:
            tds = "".join([f"<td>{self._esc(str(r.get(c,'')))}</td>" for c in cols])
            body_rows.append(f"<tr>{tds}</tr>")
        body = "\n".join(body_rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def _render_warnings(self, warnings: Any) -> str:
        if not warnings:
            return ""
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
        return " ".join([f'<span class="badge">{self._esc(w)}</span>' for w in warnings[:6]])

    def _esc(self, s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _empty_dashboard(self, message: str) -> str:
        return f"""
        <html>
          <body style="font-family:Arial;padding:20px">
            <h3>Dashboard unavailable</h3>
            <p>{self._esc(message)}</p>
          </body>
        </html>
        """