"""Build the interactive Male CNS streams page (counterpart of layering/tools/build_fly_html.py).

Injects male_cns/results/streams/artifact_data.json into malecns_streams_page.html and checks
that the static header numbers equal the embedded data.

    $PY male_cns/streams/build_malecns_html.py
Output: male_cns/outputs/streams/malecns_streams.html
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).resolve().parent / "malecns_streams_page.html"
DATA = ROOT / "male_cns/results/streams/artifact_data.json"
OUT = ROOT / "male_cns/outputs/streams/malecns_streams.html"


def main():
    html = TEMPLATE.read_text(encoding="utf-8")
    raw = DATA.read_text(encoding="utf-8")
    d = json.loads(raw)
    header = {k: re.search(rf'id="hd-{k}">([^<]+)<', html).group(1) for k in ("n", "m", "ff")}
    assert header["n"] == f"{d['n']:,}", header
    assert header["m"] == f"{d['m']:,}", header
    assert header["ff"] == f"{d['champion_ff_pct']:.2f}", header
    assert f"({d['n']:,} dots)" in html
    raw = raw.replace("</", "<\\/")
    assert html.count("__DATA__") == 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html.replace("__DATA__", raw), encoding="utf-8")
    print(f"header matches data: n={header['n']} m={header['m']} ff={header['ff']}")
    print(f"wrote {OUT} ({round(OUT.stat().st_size / 1e6, 1)} MB)")


if __name__ == "__main__":
    main()
