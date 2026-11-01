"""Build the interactive fly-connectome artifact page.

Injects outputs/fly_artifact/artifact_data.json into the template
tools/fly_streams_page.html -> outputs/fly_artifact/fly_streams.html.
The HTML is a build artifact; the template + this script are the source.

Run:  $PY layering/tools/build_fly_html.py
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "fly_streams_page.html"
DATA = HERE.parent / "outputs" / "fly_artifact" / "artifact_data.json"
OUT = HERE.parent / "outputs" / "fly_artifact" / "fly_streams.html"


def main():
    html = TEMPLATE.read_text(encoding="utf-8")
    data = DATA.read_text(encoding="utf-8")
    # keep the inline <script type="application/json"> unbreakable
    data = data.replace("</", "<\\/")
    assert "__DATA__" in html
    OUT.write_text(html.replace("__DATA__", data), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
