"""Build the interactive explorer (single HTML file) from outputs/gallery/layout.json.

    /c/ProgramData/anaconda3/python.exe layering/tools/build_html.py

The page embeds the layout JSON, loads d3 from cdnjs, and lets a reader hover /
pin a node to see its edges, switch between the two layerings, hide edge
classes and thin the drawing by weight. It is the "eyes" tool for ranking the
representations (layering/GALLERY_REVIEW.md); it reports nothing new.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GALLERY = HERE.parent / "outputs" / "gallery"

TEMPLATE = r"""<title>Mouse Connectome Layers</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#f1f0ec; --panel:#ffffff; --ink:#1f2a44; --muted:#6b7385; --line:#d9d6cf;
  --ff:#5b7fa6; --fb:#c23b3b; --intra:#e08a2e; --hub:#e6a100; --peri:#8a94a6; --core:#1f2a44;
  --accent:#2f4f7f; --shade:#ece4d8;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#151a22; --panel:#1c2230; --ink:#e6e4df; --muted:#9aa3b5; --line:#2e3646;
    --ff:#7d9fc4; --fb:#e0625f; --intra:#f0a24a; --hub:#f2b632; --peri:#6f7889; --core:#dfe3ea;
    --accent:#9db8dc; --shade:#262c38;
  }
}
:root[data-theme="dark"]{
  --bg:#151a22; --panel:#1c2230; --ink:#e6e4df; --muted:#9aa3b5; --line:#2e3646;
  --ff:#7d9fc4; --fb:#e0625f; --intra:#f0a24a; --hub:#f2b632; --peri:#6f7889; --core:#dfe3ea;
  --accent:#9db8dc; --shade:#262c38;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:baseline;gap:22px;padding:12px 18px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:17px;font-weight:600;margin:0;letter-spacing:.01em}
header .fact{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted)}
header .fact b{color:var(--ink);font-weight:500}
main{flex:1;display:flex;min-height:0}
aside{width:270px;border-right:1px solid var(--line);padding:14px 16px;display:flex;flex-direction:column;gap:14px;overflow:auto;background:var(--panel)}
aside h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 6px;font-weight:600}
.ctl{display:flex;flex-direction:column;gap:6px}
.ctl label{display:flex;align-items:center;gap:8px;cursor:pointer}
.ctl input[type=range]{width:100%}
.sw{width:18px;height:3px;border-radius:2px;display:inline-block}
select,input[type=text]{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink);width:100%}
select:focus,input:focus,button:focus{outline:2px solid var(--accent);outline-offset:1px}
.card{border-top:1px solid var(--line);padding-top:10px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-variant-numeric:tabular-nums}
.card .k{color:var(--muted)}
.card .h{grid-column:1/-1;font-family:"IBM Plex Sans",system-ui,sans-serif;font-weight:600;font-size:13px;margin-bottom:2px}
.hint{color:var(--muted);font-size:12px}
#stage{flex:1;min-width:0;position:relative}
svg{width:100%;height:100%;display:block}
.lyr-label{font:11px "IBM Plex Mono",ui-monospace,monospace;fill:var(--muted)}
.node{stroke:var(--panel);stroke-width:1.2;cursor:pointer}
.node.dim{opacity:.18}
.node.hi{stroke:var(--ink);stroke-width:2}
.lab{font:8px "IBM Plex Mono",ui-monospace,monospace;fill:#fff;pointer-events:none;text-anchor:middle;dominant-baseline:central}
.lab.dim{opacity:.18}
.e{fill:none}
.e.dim{opacity:.04}
.e.hi{opacity:.95}
.legend{position:absolute;right:14px;top:10px;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:8px 10px;font-size:12px;display:grid;grid-template-columns:auto auto;gap:4px 8px;align-items:center}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
@media (prefers-reduced-motion: no-preference){ .node,.e,.lab{transition:opacity .12s} }
</style>

<header>
  <h1>Mouse connectome, MFAS-ordered, drawn in layers</h1>
  <span class="fact">nodes <b id="f-n"></b></span>
  <span class="fact">edges <b id="f-m"></b></span>
  <span class="fact">champion feedforward <b id="f-ff"></b>%</span>
  <span class="fact">order = H63 champion (pinned, oracle-verified)</span>
</header>
<main>
  <aside>
    <div class="ctl">
      <h2>Layering</h2>
      <select id="layout">
        <option value="soft">Soft slices of pi, L=8 + hub</option>
        <option value="core_periphery">Core–periphery (SCC layered, periphery at the ends)</option>
      </select>
      <span class="hint" id="layout-hint"></span>
    </div>
    <div class="ctl">
      <h2>Edge classes (by layer)</h2>
      <label><input type="checkbox" id="show-ff" checked><span class="sw" style="background:var(--ff)"></span>feedforward</label>
      <label><input type="checkbox" id="show-fb" checked><span class="sw" style="background:var(--fb)"></span>feedback</label>
      <label><input type="checkbox" id="show-intra" checked><span class="sw" style="background:var(--intra)"></span>intra-layer</label>
    </div>
    <div class="ctl">
      <h2>Hide edges lighter than</h2>
      <input type="range" id="wmin" min="0" max="100" value="0">
      <span class="hint" id="wmin-val"></span>
    </div>
    <div class="ctl">
      <h2>Find node id</h2>
      <input type="text" id="find" placeholder="e.g. 92" inputmode="numeric">
      <span class="hint">Hover a node to see its edges; click to pin, click again to release.</span>
    </div>
    <div class="card" id="card"><div class="h">No node selected</div><span class="k">hub</span><span>id 92, 40.5% of feedback weight</span></div>
  </aside>
  <div id="stage">
    <svg id="svg" role="img" aria-label="Layered drawing of the mouse connectome"></svg>
    <div class="legend">
      <span class="dot" style="background:var(--core)"></span><span>in the giant SCC (102)</span>
      <span class="dot" style="background:var(--peri)"></span><span>periphery (pure DAG)</span>
      <span class="dot" style="background:var(--hub)"></span><span>hub</span>
    </div>
  </div>
</main>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById('data').textContent);
  const N = D.nodes, E = D.edges;
  document.getElementById('f-n').textContent = N.length;
  document.getElementById('f-m').textContent = E.length;
  document.getElementById('f-ff').textContent = D.champion_ff_pct.toFixed(2);
  const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
  const wmax = d3.max(E, e => e.w), wmin = d3.min(E, e => e.w);
  const lwScale = w => 0.4 + 3.2 * Math.sqrt((w - wmin) / (wmax - wmin));
  const wq = d3.scaleLinear().domain([0, 100]).range([0, d3.quantile(E.map(e => e.w).sort(d3.ascending), 0.95)]);

  const svg = d3.select('#svg');
  const gEdges = svg.append('g'), gNodes = svg.append('g'), gLabels = svg.append('g'), gLayer = svg.append('g');
  let pinned = null, cur = 'soft';
  const hints = {soft: 'Exact DP: 8 contiguous slices of the champion order, hub forced into its own layer. 4.1% of weight stays intra-layer.',
                 core_periphery: 'Column 0 = periphery sources, last column = periphery sinks; the 102-node SCC is soft-layered in between, hub alone.'};

  function layout(){
    const L = D.layouts[cur];
    const nL = L.widths.length, maxW = d3.max(L.widths);
    const W = 150 * nL + 140, H = 15 * maxW + 90;
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('preserveAspectRatio', 'xMidYMid meet');
    const x = d3.scaleLinear().domain([0, nL - 1]).range([70, W - 70]);
    const y = d3.scaleLinear().domain([-maxW / 2, maxW / 2]).range([H - 30, 40]);
    N.forEach((d, i) => { d.layer = L.layer[i]; d.px = x(L.layer[i]); d.py = y(L.y[i]); });
    E.forEach((e, i) => { e.cls = L.cls[i]; });
    document.getElementById('layout-hint').textContent = hints[cur];

    gLayer.selectAll('text').data(L.widths).join('text').attr('class', 'lyr-label')
      .attr('x', (d, i) => x(i)).attr('y', 22).attr('text-anchor', 'middle')
      .text((d, i) => `L${i} · ${d}`);

    gEdges.selectAll('path').data(E).join('path')
      .attr('class', 'e')
      .attr('stroke', e => e.cls === 1 ? css('--ff') : e.cls === -1 ? css('--fb') : css('--intra'))
      .attr('stroke-width', e => lwScale(e.w))
      .attr('d', e => {
        const a = N[e.s], b = N[e.t];
        if (e.cls === 1) return `M${a.px},${a.py}L${b.px},${b.py}`;
        const dx = b.px - a.px, dy = b.py - a.py;
        const bend = e.cls === -1 ? 0.25 : 0.6;
        const cx = (a.px + b.px) / 2 - dy * bend, cy = (a.py + b.py) / 2 + dx * bend;
        return `M${a.px},${a.py}Q${cx},${cy} ${b.px},${b.py}`;
      })
      .attr('marker-end', e => e.cls === 1 ? null : 'url(#arr-' + (e.cls === -1 ? 'fb' : 'intra') + ')');

    gNodes.selectAll('circle').data(N).join('circle').attr('class', 'node')
      .attr('r', d => d.hub ? 11 : 7).attr('cx', d => d.px).attr('cy', d => d.py)
      .attr('fill', d => d.hub ? css('--hub') : d.scc ? css('--core') : css('--peri'))
      .on('mouseenter', (ev, d) => { if (!pinned) focus(d); })
      .on('mouseleave', () => { if (!pinned) focus(null); })
      .on('click', (ev, d) => { pinned = pinned === d ? null : d; focus(pinned); });

    gLabels.selectAll('text').data(N).join('text').attr('class', 'lab')
      .attr('x', d => d.px).attr('y', d => d.py).text(d => d.id)
      .attr('fill', d => d.hub ? '#000' : '#fff');
    applyFilters();
  }

  const defs = svg.append('defs');
  [['fb', '--fb'], ['intra', '--intra']].forEach(([k, v]) => {
    defs.append('marker').attr('id', 'arr-' + k).attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 5).attr('markerHeight', 5).attr('orient', 'auto-start-reverse')
      .append('path').attr('d', 'M0,0L10,5L0,10z').attr('fill', css(v));
  });

  function visible(e){
    const on = e.cls === 1 ? document.getElementById('show-ff').checked
             : e.cls === -1 ? document.getElementById('show-fb').checked
             : document.getElementById('show-intra').checked;
    return on && e.w >= wq(+document.getElementById('wmin').value);
  }
  function applyFilters(){
    gEdges.selectAll('path').attr('display', e => visible(e) ? null : 'none');
    document.getElementById('wmin-val').textContent = 'w ≥ ' + wq(+document.getElementById('wmin').value).toFixed(4);
    focus(pinned);
  }
  function focus(d){
    const es = gEdges.selectAll('path'), ns = gNodes.selectAll('circle'), ls = gLabels.selectAll('text');
    if (!d) { es.classed('dim', false).classed('hi', false); ns.classed('dim', false).classed('hi', false); ls.classed('dim', false); card(null); return; }
    const nb = new Set([d.i]);
    E.forEach(e => { if (e.s === d.i) nb.add(e.t); if (e.t === d.i) nb.add(e.s); });
    es.classed('hi', e => e.s === d.i || e.t === d.i).classed('dim', e => !(e.s === d.i || e.t === d.i));
    ns.classed('hi', n => n.i === d.i).classed('dim', n => !nb.has(n.i));
    ls.classed('dim', n => !nb.has(n.i));
    card(d);
  }
  function card(d){
    const c = document.getElementById('card');
    if (!d) { c.innerHTML = '<div class="h">No node selected</div><span class="k">hub</span><span>id 92, 40.5% of feedback weight</span>'; return; }
    let outFF = 0, outFB = 0, inFF = 0, inFB = 0, nOut = 0, nIn = 0;
    E.forEach(e => {
      if (e.s === d.i) { nOut++; if (e.ff_pi) outFF += e.w; else outFB += e.w; }
      if (e.t === d.i) { nIn++; if (e.ff_pi) inFF += e.w; else inFB += e.w; }
    });
    const f = v => v.toFixed(4);
    c.innerHTML = `<div class="h">node id ${d.id}${d.hub ? ' · hub' : ''}</div>
      <span class="k">pi rank</span><span>${d.rank} / ${N.length}</span>
      <span class="k">layer</span><span>L${d.layer}</span>
      <span class="k">membership</span><span>${d.scc ? 'giant SCC' : 'periphery'}</span>
      <span class="k">out edges</span><span>${nOut} · ff ${f(outFF)} · fb ${f(outFB)}</span>
      <span class="k">in edges</span><span>${nIn} · ff ${f(inFF)} · fb ${f(inFB)}</span>`;
  }

  document.getElementById('layout').addEventListener('change', ev => { cur = ev.target.value; pinned = null; layout(); });
  ['show-ff', 'show-fb', 'show-intra', 'wmin'].forEach(id => document.getElementById(id).addEventListener('input', applyFilters));
  document.getElementById('find').addEventListener('input', ev => {
    const v = +ev.target.value; const d = N.find(n => n.id === v);
    pinned = d || null; focus(pinned);
  });
  window.addEventListener('resize', layout);
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', layout);
  layout();
})();
</script>
"""


def main() -> None:
    data = json.loads((GALLERY / "layout.json").read_text(encoding="utf-8"))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = GALLERY / "explorer.html"
    out.write_text(html, encoding="utf-8")
    print("written", out, f"{out.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
