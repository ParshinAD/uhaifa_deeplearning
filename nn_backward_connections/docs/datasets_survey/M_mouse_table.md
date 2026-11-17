# Provenance of `data/table_mouse.txt` (follow-up to C_mesoscale.md "Side task: provenance")

Date: 2026-11-17. The data file was not modified.

## Fingerprint of our file
Tab-separated `src tgt weight`, no header. 148 nodes (ids 1..149 without 37), 583 directed edges, no self-loops, no duplicates. Weights 0.000184 to 0.614, median about 0.0063, sum 9.159036. Max in-degree 95 and max out-degree 56, both on node 92. Next in-degrees 40, 25, 24, 23; next out-degrees 24, 13, 13. Heavy edges: 51->53 0.613959345844929, 105->123 0.594344938173106.

Already ruled out earlier (not redone): Oh 2014 linear model from Netzschleuder, Harris 2019 cortical (43 areas), SEU-net 2025 with layers, Calabrese 2015 dMRI.

## Method
For each candidate matrix (and its transpose) we computed:
- node count, nonzero count, max in/out degree, and max in/out degree after keeping the top 583 weights;
- exact weight matches: how many of our 583 weights have a candidate value within relative 1e-8;
- scale test: for our 30 largest weights, the log-offset to every candidate value, rounded to 1e-7; the most common offset and how many of the 30 weights support it. A real match up to a constant factor gives 30 of 30;
- heavy-edge ratio: number of candidate value pairs whose ratio equals 0.613959345844929 / 0.594344938173106 within 1e-9 in log.

Positive control: our 583 weights planted (times 3.7) into a random dense 291x291 lognormal matrix. Result: scale support 30 of 30; without scaling, 583 of 583 exact matches. So the tests detect a true source, also after a constant factor.

Chance level in large dense matrices (84k values): up to about 1 exact match at 1e-8 and scale support 3 to 5.

## Candidates checked

1. SEU-net paper (Xiong et al. 2025, Nat Methods), Zenodo record 17182254, `jupyter_tutorial.zip` (975,605,622 bytes). Notebook `03_connectome_comparison.ipynb` shows that "Allen-net" is simply Oh 2014 Suppl. Table (file `41586_2014_BFnature13186_MOESM71_ESM.xlsx`) W_ipsi + W_contra on 213 regions, and "BRICseq-net" is `BRICseq_connection_strength.csv` (38 x 41 cortical areas).

| Matrix | Shape | Nonzero | Max in / out | Exact 1e-8 | Scale support (of 30) |
|---|---|---|---|---|---|
| BRICseq_connection_strength | 38x41 | 1,519 | 38 / 41 | 0 | 2 |
| D18_SEU_net | 90x223 | 2,332 | 64 / 107 | 0 | 2 |
| SEU_net_wo_corticalLayers | 90x90 | 922 | 45 / 38 | 0 | 2 |
| SEU_net_w_corticalLayers | 148x148 | 1,365 | 66 / 55 | 0 | 2 |
| Oh 2014 W_ipsi (original xlsx) | 213x213 | 16,954 | 115 / 182 | 0 | 3 |
| Oh 2014 W_contra | 213x213 | 15,868 | 111 / 191 | 0 | 3 |
| Allen-net (W_ipsi + W_contra) | 213x213 | 21,807 | 139 / 198 | 0 | 3 |

BRICseq has only 41 nodes, too few for our 148. Transposes give the same counts.

2. Oh 2014 variants from the original xlsx: W_ipsi, W_contra and Allen-net each with self-loops removed and normalized by row sum, column sum, max and total; also W_ipsi and W_contra kept only where PValue < 0.05 or < 0.01 (3,063 / 2,431 ipsi and 2,353 / 1,885 contra nonzero), with the same four normalizations. Best result over all 28 variants: 1 exact match, scale support 3. Top-583 max in-degree at most 52.

3. Knox et al. 2019 regionalized matrices (download.alleninstitute.org, "A_high_resolution_data-driven_model_of_the_mouse_connectome"): connection_density, connection_strength, normalized_connection_density, normalized_connection_strength (291 sources x 577 targets, ipsi and contra), and normalized_connection_density_ipsi_ctx (43x43). Each tested raw, ipsi only, ipsi transposed, and ipsi/contra with row, column, max and total normalization. Best result: 1 exact match, scale support 5 (chance level). The matrices are almost fully dense (84k of 84.7k ipsi entries nonzero).
   One near-coincidence: in normalized_connection_strength ipsi, the top 583 edges give a hub with in-degree 94 (CP, Caudoputamen, Allen id 672). But that subgraph has max out-degree 12 (ours 56 on the same hub), touches 193 nodes (ours 148), contains 80 self-loops (ours 0), and no weights match. So it is not our table. It does suggest our node 92 may be a striatum-like target, which is a guess only.

4. Web search for the filename "table_mouse.txt" and for "148 regions, 583 edges": no hit (2 searches).

## Verdict
Ruled out: Allen-net and BRICseq-net of the SEU-net paper, all SEU-net variants in the tutorial zip, Oh 2014 W_ipsi / W_contra / ipsi+contra with p-value thresholds and simple normalizations, and the Knox 2019 regionalized matrices in all simple normalizations. No candidate has a single non-chance weight match, at exact precision or after a constant factor. Provenance is still unknown.

## Remaining leads
- The file may be a derived product: weights after a nonlinear transform (log, rank, fraction of injection volume per experiment), or values from a single experiment or a model fit we do not have. Exact-match tests can not catch nonlinear transforms; a rank-correlation test needs a node mapping first.
- Other mouse sources with about 150 regions: Zingg 2014 / Mouse Connectome Project cortical matrices, Wang et al. 2022 chemo-connectome (295 areas, figshare 20033399), BAMS-style collations, Rubinov 2015 processed versions.
- Git history gives nothing: `git log --follow data/table_mouse.txt` shows only commit 7a77547 (2026-06-21, "main reproduce rocket algorithm"), where the file was added together with the rest of the project. Asking whoever supplied the file (course materials, supervisor, challenge organizers) is probably the cheapest remaining lead.

## Reproduce
Scripts were run with `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python -I` from a scratch directory, not kept in the repo. openpyxl is not installed in `allen`; the xlsx was read with a small stdlib (zipfile + xml) reader. Downloads were deleted after the check.

## Sources
- https://zenodo.org/records/17182254 (jupyter_tutorial.zip, SEU_net_*.xlsx)
- https://www.nature.com/articles/s41592-025-02784-2
- https://www.nature.com/articles/nature13186 (Suppl. MOESM71)
- https://download.alleninstitute.org/publications/A_high_resolution_data-driven_model_of_the_mouse_connectome/
- https://direct.mit.edu/netn/article/3/1/217/2194 (Knox et al. 2019)
