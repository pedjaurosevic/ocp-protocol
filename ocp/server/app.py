"""
OCP Leaderboard Server — SQLite-backed, FastAPI.

Run with:  ocp serve
or:        uvicorn ocp.server.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ocp.server.db import LeaderboardDB

app = FastAPI(
    title="OCP Leaderboard",
    description="Open Consciousness Protocol — public model leaderboard",
    version="0.1.0",
)

DB_PATH = Path.home() / ".ocp" / "leaderboard.db"
_db: Optional[LeaderboardDB] = None


def get_db() -> LeaderboardDB:
    global _db
    if _db is None:
        _db = LeaderboardDB(DB_PATH)
    return _db


# ── Models ────────────────────────────────────────────────────────────────────

class SubmitResult(BaseModel):
    """Payload for POST /results — full EvaluationResult JSON."""
    result: dict
    submitter: Optional[str] = None
    notes: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve embedded leaderboard HTML."""
    db = get_db()
    rows = db.get_leaderboard(limit=50)
    return _render_leaderboard_html(rows)


@app.get("/api/leaderboard")
async def api_leaderboard(
    limit: int = Query(50, ge=1, le=200),
    min_level: int = Query(1, ge=1, le=5),
):
    db = get_db()
    rows = db.get_leaderboard(limit=limit, min_level=min_level)
    return {"count": len(rows), "results": rows}


@app.post("/api/results", status_code=201)
async def submit_result(payload: SubmitResult):
    """Submit an evaluation result to the leaderboard."""
    db = get_db()
    result_id = db.insert_result(
        result=payload.result,
        submitter=payload.submitter,
        notes=payload.notes,
    )
    return {"id": result_id, "message": "Result submitted successfully"}


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    db = get_db()
    row = db.get_result(result_id)
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")
    return row


@app.get("/api/stats")
async def stats():
    db = get_db()
    return db.get_stats()


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time(), "version": "0.1.0"}


# ── HTML Renderer ─────────────────────────────────────────────────────────────

def _render_leaderboard_html(rows: list[dict]) -> str:
    level_colors = {
        1: "#8b949e", 2: "#d29922", 3: "#3fb950", 4: "#58a6ff", 5: "#bc8cff"
    }

    def bar(v: float | None, w: int = 80) -> str:
        if v is None:
            return '<span style="color:#8b949e">—</span>'
        pct = round(max(0, min(1, v)) * w)
        fill = f'<div style="background:#58a6ff;height:6px;border-radius:3px;width:{pct}px;"></div>'
        return (f'<div style="display:inline-flex;align-items:center;gap:6px;">'
                f'<div style="background:#21262d;border-radius:3px;height:6px;'
                f'width:{w}px;display:inline-block;">{fill}</div>'
                f'<span style="font-family:monospace;font-size:11px;">{v:.3f}</span></div>')

    rows_html = ""
    for i, r in enumerate(rows, 1):
        lvl = r.get("ocp_level") or 1
        color = level_colors.get(lvl, "#8b949e")
        badge = f'<span style="color:{color};font-weight:bold;">OCP-{lvl}</span>'
        model = f"{r['provider']}/{r['model']}"
        sasmi = r.get("sasmi_score")
        date = time.strftime("%Y-%m-%d", time.localtime(r.get("timestamp", 0)))
        rows_html += f"""<tr>
          <td style="color:#8b949e;font-family:monospace;">{i}</td>
          <td style="color:#58a6ff;font-weight:bold;">{model}</td>
          <td>{badge} <span style="color:#8b949e;font-size:11px;">{r.get('ocp_level_name','')}</span></td>
          <td>{bar(sasmi)}</td>
          <td style="color:#8b949e;font-size:11px;">{date}</td>
        </tr>"""

    count = len(rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCP Leaderboard</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0d1117; color:#e6edf3; font-family:'Segoe UI',system-ui,sans-serif;
          padding:2rem; line-height:1.6; }}
  h1 {{ color:#58a6ff; font-size:1.8rem; margin-bottom:.25rem; }}
  .sub {{ color:#8b949e; font-size:.9rem; margin-bottom:2rem; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ color:#8b949e; text-align:left; font-size:.8rem; text-transform:uppercase;
        letter-spacing:.08em; padding:.5rem .75rem; border-bottom:1px solid #30363d; }}
  td {{ padding:.5rem .75rem; border-bottom:1px solid #21262d; font-size:.9rem; }}
  tr:hover td {{ background:#161b22; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
           padding:1.25rem; margin-bottom:1.5rem; }}
  .api {{ font-size:.8rem; color:#8b949e; margin-top:2rem; border-top:1px solid #30363d;
          padding-top:1rem; }}
  code {{ background:#21262d; padding:.1em .4em; border-radius:3px;
          font-family:monospace; font-size:.85em; }}
</style>
</head>
<body>
<h1>OCP Leaderboard</h1>
<p class="sub">Open Consciousness Protocol v0.1.0 · {count} result{"s" if count != 1 else ""}</p>

<div class="card">
  <table>
    <thead><tr>
      <th>#</th><th>Model</th><th>OCP Level</th><th>SASMI</th><th>Date</th>
    </tr></thead>
    <tbody>{rows_html if rows_html else
      '<tr><td colspan="5" style="color:#8b949e;text-align:center;padding:2rem;">No results yet. Submit with <code>ocp submit</code></td></tr>'}</tbody>
  </table>
</div>

<div class="api">
  <strong>API endpoints:</strong>
  <code>GET /api/leaderboard</code> ·
  <code>POST /api/results</code> ·
  <code>GET /api/results/{{id}}</code> ·
  <code>GET /api/stats</code>
  <br><br>Submit a result: <code>ocp submit --results path/to/result.json --server http://localhost:8080</code>
</div>
</body>
</html>"""
