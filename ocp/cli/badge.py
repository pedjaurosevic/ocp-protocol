"""
OCP Badge Generator — produces an SVG badge suitable for README / HuggingFace model cards.

Format:
  [ OCP | OCP-3 · SASMI 0.62 · v0.1.0 ]
"""

from __future__ import annotations

import json
import time
from pathlib import Path


# Color palette per OCP level
LEVEL_COLORS = {
    1: ("#555", "#aaa"),       # gray
    2: ("#555", "#d29922"),    # yellow
    3: ("#555", "#3fb950"),    # green
    4: ("#555", "#58a6ff"),    # blue
    5: ("#555", "#bc8cff"),    # purple
}

BADGE_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="20">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{width}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{left_w}" height="20" fill="{left_bg}"/>
    <rect x="{left_w}" width="{right_w}" height="20" fill="{right_bg}"/>
    <rect width="{width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{left_cx}" y="15" fill="#010101" fill-opacity=".3">{left_text}</text>
    <text x="{left_cx}" y="14">{left_text}</text>
    <text x="{right_cx}" y="15" fill="#010101" fill-opacity=".3">{right_text}</text>
    <text x="{right_cx}" y="14">{right_text}</text>
  </g>
</svg>"""


def generate_badge(results_path: str | Path, output_path: str | Path) -> Path:
    """Generate an SVG badge from a results JSON file."""
    data = json.loads(Path(results_path).read_text())

    ocp_level = data.get("ocp_level", 1)
    level_name = data.get("ocp_level_name", "")
    sasmi = data.get("sasmi_score")
    version = data.get("protocol_version", "0.1.0")
    date = time.strftime("%Y-%m-%d", time.localtime(data.get("timestamp", time.time())))

    left_text = "OCP"
    sasmi_str = f"SASMI {sasmi:.2f}" if sasmi is not None else "no SASMI"
    right_text = f"OCP-{ocp_level} · {sasmi_str} · v{version}"

    left_bg, right_bg = LEVEL_COLORS.get(ocp_level, ("#555", "#aaa"))

    char_w = 6.5
    pad = 10
    left_w = int(len(left_text) * char_w + pad * 2)
    right_w = int(len(right_text) * char_w + pad * 2)
    width = left_w + right_w

    svg = BADGE_SVG.format(
        width=width,
        left_w=left_w,
        right_w=right_w,
        left_bg=left_bg,
        right_bg=right_bg,
        left_cx=left_w // 2,
        right_cx=left_w + right_w // 2,
        left_text=left_text,
        right_text=right_text,
    )

    out = Path(output_path)
    out.write_text(svg)
    return out
