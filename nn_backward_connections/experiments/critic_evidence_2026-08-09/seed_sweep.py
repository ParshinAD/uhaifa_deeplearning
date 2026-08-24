"""Does the seed-42 sizing number generalize? Re-run the probe's own run_dataset()
on the H35 orders for seeds 123 and 999 (already on disk). No probe code modified."""
import sys, json, time
from pathlib import Path
ROOT = Path("/Users/abed359/IdeaProjects/university/deeplearning_thesis/nn_backward_connections")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"experiments"))
import size_collective_moves as S

out = {}
for ds, seeds in (("connectome", (123, 999)), ("mouse", (123, 999))):
    for sd in seeds:
        t = time.time()
        r = S.run_dataset(ds, sd, 200000, 6)
        out[f"{ds}-s{sd}"] = {
            "h35_pct": r["h35_pct"],
            "positions": r["h35_positions_file"],
            "S1_only_pp": r["S1_paired_backward_move"]["realized_gain_pp"],
            "S2_best_pp": max((x["total_gain_pp"] for x in r["S2_block_scc"]), default=0.0),
            "iter_block_size": r["iterated_S1_S2"]["block_size"],
            "iter_total_pp": r["iterated_S1_S2"]["total_gain_pp"],
            "iter_after_pct": r["iterated_S1_S2"]["oracle_after_pct"],
            "per_round_cum_pp": [c["cumulative_gain_pp"] for c in r["iterated_S1_S2"]["curve"]],
            "iter_seconds": sum(c["seconds"] for c in r["iterated_S1_S2"]["curve"]),
        }
        print(json.dumps({f"{ds}-s{sd}": out[f"{ds}-s{sd}"]}, indent=1), flush=True)
Path("/Users/abed359/IdeaProjects/university/deeplearning_thesis/nn_backward_connections/dr_tmp/critic_collective/seed_sweep.json").write_text(json.dumps(out, indent=2))
print("DONE")
