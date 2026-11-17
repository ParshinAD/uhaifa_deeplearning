"""Build the interactive fly-connectome artifact page.

Injects into the template tools/fly_streams_page.html:
  __WHATIF_CORE__  <- tools/fly_whatif_core.js (the what-if code the Node test also runs)
  __EDGES__        <- base64 of outputs/fly_artifact/edges.zlib (from fly_artifact_export.py)
  __DATA__         <- outputs/fly_artifact/artifact_data.json
-> outputs/fly_artifact/fly_streams.html, one self-contained file.
The HTML is a build artifact; the template, the core file + this script are the source.
Design: layering/VIZ_V2_DESIGN.md section 2.3.

Run:  $PY layering/tools/build_fly_html.py
"""
from __future__ import annotations

import base64
from pathlib import Path

CONFIG = {"max_html_bytes": 19 * 2**20}   # artifact limit is 20 MiB

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "fly_streams_page.html"
CORE = HERE / "fly_whatif_core.js"
DATA = HERE.parent / "outputs" / "fly_artifact" / "artifact_data.json"
EDGES = HERE.parent / "outputs" / "fly_artifact" / "edges.zlib"
OUT = HERE.parent / "outputs" / "fly_artifact" / "fly_streams.html"


def main():
    """Fill the three placeholders (core and edges before data) and check the size."""
    html = TEMPLATE.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    edges = base64.b64encode(EDGES.read_bytes()).decode("ascii")
    data = DATA.read_text(encoding="utf-8")
    # keep the inline <script type="application/json"> unbreakable
    data = data.replace("</", "<\\/")
    assert "</script" not in core.lower()
    assert "__EDGES__" not in core and "__DATA__" not in core, "core must not contain a placeholder"
    for ph in ("__WHATIF_CORE__", "__EDGES__", "__DATA__"):
        assert html.count(ph) == 1, f"{ph} must occur exactly once in the template"
    # data last, so data text can never contain a placeholder that gets replaced
    html = html.replace("__WHATIF_CORE__", core).replace("__EDGES__", edges).replace("__DATA__", data)
    out = html.encode("utf-8")
    size = len(out)
    assert size < CONFIG["max_html_bytes"], f"page would be {size:,} bytes, over the build size cap; nothing written"
    OUT.write_bytes(out)
    print(f"wrote {OUT}  ({size:,} bytes = {round(size / 2**20, 2)} MiB, cap {CONFIG['max_html_bytes']:,})")


if __name__ == "__main__":
    main()
