# Male CNS streams page

The interactive "streams" page of the FlyWire connectome (`layering/outputs/fly_artifact/fly_streams.html`),
rebuilt for the Male CNS v1.0 graph and its one-seed H64 order. Same views (Streams, Matrix,
Chemistry), same controls, same layout and statistics code: the FlyWire functions are imported
from `layering/tools/`, nothing there was changed.

## Files

| file | what |
|---|---|
| `malecns_streams_prep.py` | loads `data/processed/malecns.npz` and `male_cns/results/malecns_H64_s42_rank.npy`, re-scores the order with the frozen oracle (asserts 93,041,515), encodes the annotation columns of `data/processed/malecns_ann.csv.gz`, computes the spectral coordinate with `fly_ann_prep.spectral_coords` -> `results/streams/node_ann.npz`, `node_ann_meta.json` |
| `malecns_streams.py` | lanes (the `lane` column, ordered by median rank), layouts for 8/16/32 bins via `fly_streams.build_layout` and `layout_stats`, flow test, NT x direction, sink audit, lane matrix -> `results/streams/layout_*.json`, `lane_matrix.json`, `report_streams.json`; static previews `outputs/streams/S1_malecns_streams_mcns21_B{8,16,32}.png`. Holds the lane colours. |
| `malecns_artifact_export.py` | the page data (same schema as `fly_artifact_export.py`, plus `fw_lane_median_frac` and `compare`) -> `results/streams/artifact_data.json` |
| `malecns_streams_page.html` | page template, a copy of `layering/tools/fly_streams_page.html` with Male CNS text |
| `build_malecns_html.py` | injects the data into the template, asserts the header numbers equal the data -> `outputs/streams/malecns_streams.html` |

## Run

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
$PY male_cns/streams/malecns_streams_prep.py       # 43 s (spectral coordinate 42 s)
$PY male_cns/streams/malecns_streams.py            # 23 s (incl. 2,000-permutation flow test, 3 PNGs)
$PY male_cns/streams/malecns_artifact_export.py    # 5 s
$PY male_cns/streams/build_malecns_html.py         # < 1 s
```

Runtimes measured on the MacBook (2026-11-17), logs in `male_cns/logs/streams_*.log`.
The page is one self-contained file of 5.3 MB (fonts load from Google Fonts when online).

## Differences from the FlyWire pipeline

- Graph and order: Male CNS brain + nerve cord, weight >= 2, traced only; order = H64 pipeline,
  one seed (42), Apple MPS, 81.8079 % FF. FlyWire uses the pinned CUDA champion (84.2582 %) via
  `layering/io_utils.py`; the male order is not in `sota.json`, so the check here is an exact
  re-score against the logged FF weight instead.
- Lanes: 21 instead of 17. They come ready-made from `male_cns/pipeline/ann_prep.py`
  (optic lanes through the FlyWire class of the matched `flywireType`). New lanes: VNC sensory,
  VNC intrinsic, motor (VNC), motor (brain) (FlyWire "motor", same colour), unannotated.
  Shared lanes keep the FlyWire colour; new colours were picked for CIELAB distance >= 15 from
  every other lane colour and the sink hatch, none from the red family.
- Raw groupings: Male CNS `class` (22 values incl. NA) and `superclass` (26). They are stored
  under the FlyWire keys `cell_class` / `super_class` so the page code did not change.
- Transmitters: Male CNS `consensus_nt`, short codes mapped back to the FlyWire long names;
  histamine is new (photoreceptors), "unclear" counts as NA, Kenyon cells set to ACh (as FlyWire).
  The page got a histamine colour and bar.
- Flow: from the Male CNS superclass (`ann_prep.py`); descending neurons count as intrinsic.
  The flow rho uses the FlyWire function (stable tie-break), 0.237 here vs 0.263 FlyWire. It is
  not comparable with `analyze.py`'s 0.470 (average ranks for ties).
- Hemisphere: Male CNS `somaSide`.
- Cell types: Male CNS `type`; 2,069 untyped neurons are shown as "<class> (class)".
- Page additions: lane label, lane hover card, cell card and regions panel show the lane median as
  a fraction of the order next to the FlyWire median (`male_cns/results/analysis_flywire.json`,
  FlyWire "motor" for "motor (brain)"). The methods panel fills its numbers from the embedded
  data and has a short FlyWire vs Male CNS table (from `analysis_*.json`, `vnc_loop.json`).
- Not ported: `fly_cloud.py` static cloud PNGs (the page computes the neuron layer from the
  export, as on FlyWire), `fly_streams_proposal.py`.
