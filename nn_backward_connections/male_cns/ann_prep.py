"""Per-node annotations of the Male CNS graph in the vocabulary of the FlyWire analyses.

Male CNS superclasses differ from FlyWire's (it has the nerve cord, and optic-lobe intrinsic
neurons carry no neuropil class). Lanes follow layering/tools/fly_streams.py:lane_of; optic
neurons get the FlyWire cell_class of their matched flywireType (majority over FlyWire
neurons of that type), so the optic lanes use exactly the FlyWire rule. Nerve-cord neurons get
their own lanes.

    $PY male_cns/ann_prep.py
Output: data/processed/malecns_ann.csv.gz, male_cns/results/ann_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "layering" / "tools"))
from fly_streams import MB_CLASSES, OLFACTION_CLASSES  # noqa: E402

CONFIG = dict(
    nodes="data/processed/malecns_nodes.csv.gz",
    flywire_ann="data/external/flywire_neuron_annotations.tsv.gz",
    out="data/processed/malecns_ann.csv.gz",
    nt_short={"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu", "dopamine": "DA",
              "serotonin": "5-HT", "octopamine": "OA", "histamine": "HA"},
)
AFFERENT = {"cb_sensory", "ol_sensory", "vnc_sensory", "sensory_ascending", "sensory_descending",
            "cb_sensory_tbc", "vnc_sensory_tbc", "sensory_ascending_tbc"}
EFFERENT = {"cb_motor", "vnc_motor", "vnc_efferent", "cb_efferent", "efferent_ascending",
            "efferent_descending", "cb_endocrine", "vnc_endocrine"}
VNC = {"vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent", "vnc_endocrine", "vnc_tbc",
       "vnc_sensory_tbc"}


def optic_lane(cc):
    """FlyWire optic lane rule (fly_streams.lane_of, super_class == 'optic')."""
    if not isinstance(cc, str):
        return "optic (other)"
    head = cc.split(">")[0].split(".")[0]
    if head == "LA":
        return "lamina"
    if head == "ME":
        return "medulla→lobula" if (">" in cc and cc.split(">")[1][:2] == "LO") else "medulla"
    if head in ("LO", "LOP"):
        return "lobula (plate)"
    return "optic (other)"


def lane_of_malecns(sc, cls, fw_cc):
    """Map one Male CNS neuron (superclass, class, FlyWire cell_class of its type) to a lane."""
    sc = sc if isinstance(sc, str) else ""
    if sc in ("ol_sensory",) or (sc == "cb_sensory" and cls == "visual"):
        return "photoreceptors"
    if sc in ("cb_sensory", "cb_sensory_tbc", "sensory_descending"):
        return "sensory (other)"
    if sc in ("vnc_sensory", "vnc_sensory_tbc"):
        return "VNC sensory"
    if sc.startswith("ascending") or sc.startswith("sensory_ascending"):
        return "ascending (VNC)"
    if sc == "ol_intrinsic":
        return optic_lane(fw_cc)
    if sc == "visual_centrifugal":
        return "visual centrifugal"
    if sc.startswith("visual_projection"):
        return "visual projection"
    if sc == "cb_intrinsic":
        c = cls if isinstance(cls, str) else fw_cc
        if c in MB_CLASSES:
            return "mushroom body"
        if c in ("CX", "TuBu"):
            return "central complex"
        if c in OLFACTION_CLASSES:
            return "olfaction: AL·LH"
        return "central (other)"
    if sc == "descending_neuron":
        return "descending"
    if sc in ("cb_endocrine", "vnc_endocrine"):
        return "endocrine"
    if sc in ("cb_motor", "cb_efferent"):
        return "motor (brain)"
    if sc in ("vnc_motor", "vnc_efferent", "efferent_ascending", "efferent_descending"):
        return "motor (VNC)"
    if sc in ("vnc_intrinsic", "vnc_tbc"):
        return "VNC intrinsic"
    return "unannotated"


n = pd.read_csv(ROOT / CONFIG["nodes"], low_memory=False)
fw = pd.read_csv(ROOT / CONFIG["flywire_ann"], sep="\t", low_memory=False,
                 usecols=["cell_type", "cell_class"])
fw_cc = fw.dropna().groupby("cell_type").cell_class.agg(lambda s: s.mode().iat[0])
n["fw_cell_class"] = n.flywireType.map(fw_cc)
n["lane"] = [lane_of_malecns(a, b, c) for a, b, c in zip(n.superclass, n["class"], n.fw_cell_class)]
n["flow"] = np.where(n.superclass.isin(AFFERENT), "afferent",
                     np.where(n.superclass.isin(EFFERENT), "efferent", "intrinsic"))
n["region"] = np.where(n.superclass.isin(VNC), "VNC", "brain")
# Kenyon cells: predicted DA is a known synapse-classifier artefact; FlyWire analyses override to ACh.
nt = n.consensus_nt.where(~n.consensus_nt.isin(["unclear"]), np.nan)
nt = nt.where(n["class"] != "Kenyon_Cell", "acetylcholine")
n["top_nt"] = nt.map(CONFIG["nt_short"])
n["side"] = n.somaSide.map({"L": "left", "R": "right", "M": "center"}).fillna("na")
n["cell_type"] = n["type"]
cols = ["idx", "bodyId", "cell_type", "flywireType", "superclass", "class", "fw_cell_class", "lane",
        "flow", "region", "top_nt", "side", "group", "dimorphism", "fruDsx"]
n[cols].to_csv(ROOT / CONFIG["out"], index=False)
summary = dict(
    n=int(len(n)),
    lane_counts={k: int(v) for k, v in n.lane.value_counts().items()},
    flow_counts={k: int(v) for k, v in n.flow.value_counts().items()},
    region_counts={k: int(v) for k, v in n.region.value_counts().items()},
    top_nt_counts={str(k): int(v) for k, v in n.top_nt.value_counts(dropna=False).items()},
    side_counts={k: int(v) for k, v in n.side.value_counts().items()},
    optic_intrinsic_without_fw_class=int(((n.superclass == "ol_intrinsic") & n.fw_cell_class.isna()).sum()),
    kenyon_cells_nt_overridden=int((n["class"] == "Kenyon_Cell").sum()),
)
(ROOT / "male_cns" / "results" / "ann_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
