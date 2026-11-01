"""Inject the artifact data bundle into the page template.

microns_page.html + microns_artifact/artifact_data.json ->
layering/outputs/microns_artifact/microns_streams.html (the published artifact).

Run:  $PY layering/tools/build_microns_html.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "microns_page.html"
DATA = HERE.parent / "outputs" / "microns_artifact" / "artifact_data.json"
OUT = HERE.parent / "outputs" / "microns_artifact" / "microns_streams.html"

html = PAGE.read_text(encoding="utf-8")
data = DATA.read_text(encoding="utf-8")
assert "__DATA__" in html, "page template lost its __DATA__ placeholder"
# JSON inside a <script> block: escape the only dangerous sequence
data = data.replace("</", "<\\/")
OUT.write_text(html.replace("__DATA__", data), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
