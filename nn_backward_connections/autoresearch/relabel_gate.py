"""The relabelling-robustness gate (P09) — a robustness test for a DETERMINISTIC pipeline.

WRITTEN BEFORE THE DATA IT ADJUDICATES, then AMENDED by the cycle-9 critic. This module
implements the rule pre-registered in ``experiments/outputs/proto_P09_prereg.json``, which was
written before the connectome study was launched. Three of the critic's findings changed the code
rather than the prose and are marked D1/D2/D5 below.

What this is FOR — read this before reading the code
-----------------------------------------------------
This gate is an ADDITIONAL requirement on a primary promotion whose two pools are degenerate
(sample std <= ``zero_std_tol``). It does **not** replace ``require_protocol_ci_lower_gt``.

That distinction was the cycle-9 critic's central finding and it is worth stating plainly.
P09 offered the operator three ways to resolve the campaign's first-ever disagreement between its
two significance criteria, and cycle 9 initially took the one option that admitted the campaign's
own pending result. It should not have: ``autoresearch/queue.json`` P09 reserves that decision for
the operator in as many words. So the change was split in half:

* the TIGHTENING — this gate, which can only ever refuse a promotion, and which the campaign is
  free to impose on itself — is IN FORCE;
* the RETIREMENT of the PROTOCOL CI — the half that would admit a result the campaign already
  holds — is NOT, and waits for the operator.

Both gates therefore run today, and a degenerate-pool promotion must clear both.

Why a relabelling test is the right ADDITIONAL requirement
----------------------------------------------------------
``audit.py``'s PROTOCOL CI asks whether a delta exceeds ``z * max(std_comparator,
baseline_sigma_pp) * sqrt(2 / n_comparator)``. On this device the pipelines never draw from
``seed``, so the confirm pools hold ONE distinct value and there is no sampling error for a CI to
describe. Whatever one thinks of that CI, a pool with zero dispersion clearly needs SOME evidence
that the improvement is a property of the mechanism — and before P09 it needed none at all.

Relabelling supplies it. Replacing contiguous index ``i`` by ``perm[i]`` throughout yields an
ISOMORPHIC graph: the edge multiset, the weights and the set of achievable feedforward scores are
unchanged, so nothing an algorithm *ought* to do changes. Verified rather than asserted — the
cycle-9 critic mapped ``results/rocket_best_positions.npy`` through every permutation used here
and re-scored the parity anchor at exactly 34,751,902 each time.

Index-order tie-breaks do change: greedy-FAS's ordering, the sift's sweep order, and the SCC
recursion's enumeration order. (An earlier draft of this docstring also claimed stage 5's
candidate scan was perturbed. It is not — its heap key is ``(-weight, edge_index)`` and
``proto_P09_relabel.py`` deliberately leaves edge ROW order untouched, so stage 5 moves only
through its input order. D7.)

The design is PAIRED: both arms run on the same relabelled graph, so nuisance common to both
cancels. For a NESTED variant — one that is the comparator's pipeline plus a monotone stage, which
is what H52 is — the pairing is exact by construction rather than merely statistical, and
``delta_r >= 0`` holds with probability 1. **Positivity is therefore not evidence for such a pair;
only the MAGNITUDES are** (D4). Do not quote "R/R positive" as a result.

The two tests, and why there are two
-------------------------------------
1. **Min-rule.** ``delta_r > min_promotion_delta_pp`` for EVERY r. Monotone in the direction that
   cannot be gamed by re-running: adding relabellings can only make it harder to pass.
2. **Paired one-sided lower bound** (D5). ``mean(delta) - t_{1-alpha, R} * sd(delta)/sqrt(R+1) >
   min_promotion_delta_pp``.

The first draft had only the min-rule, on the argument that any CI would reintroduce the
``sigma/sqrt(n)`` manipulability that condemned the PROTOCOL CI. The critic showed that argument
conflates two different tests: a CI against ZERO is manipulable, because more samples shrink the
threshold toward the null; a CI against a FIXED non-zero effect size is consistent, because as
R grows it converges on "the true mean delta exceeds 0.012", a fixed property of the mechanism.
The min-rule has the opposite defect — it is stopping-rule sensitive, since P(some single
labelling falls below the bar) grows with R (~1.5% at R=4 rising to ~25% at R=99 for the observed
dispersion), so a campaign free to choose R has an incentive to stop at the minimum.

Requiring BOTH, at an R fixed in ``campaign.yaml`` rather than chosen per study, is what the
critic recommended and what this implements. The honest residual limitation: ``min_relabellings``
is a floor, not an exact count, so a study may still legitimately be larger than the floor. Fix R
in the pre-registration of each study.

Scope
-----
PRIMARY datasets only, and only when both pools are degenerate. A genuinely stochastic pipeline
keeps the Welch/PROTOCOL CI and this module is not consulted. ``min_promotion_delta_pp`` remains a
separate, independent requirement in all cases.

Evidence is registered explicitly in ``experiments/outputs/relabel_index.json`` rather than
discovered by glob, so an audit can never silently pick up the wrong study — and since D2 the
study's own ``variant``/``comparator``/``dataset`` fields must agree with the key it was filed
under, so a typo in the registry cannot attach the wrong one either.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = _ROOT / "experiments" / "outputs" / "relabel_index.json"

# One-sided t quantiles at 95%, indexed by degrees of freedom. A small table keeps this module
# free of a scipy import (audit.py must run anywhere the campaign runs); values agree with
# scipy.stats.t.ppf(0.95, df) to the digits shown. df >= 30 falls back to the normal quantile.
_T95 = {1: 6.313752, 2: 2.919986, 3: 2.353363, 4: 2.131847, 5: 2.015048, 6: 1.943180,
        7: 1.894579, 8: 1.859548, 9: 1.833113, 10: 1.812461, 11: 1.795885, 12: 1.782288,
        13: 1.770933, 14: 1.761310, 15: 1.753050, 16: 1.745884, 17: 1.739607, 18: 1.734064,
        19: 1.729133, 20: 1.724718, 21: 1.720743, 22: 1.717144, 23: 1.713872, 24: 1.710882,
        25: 1.708141, 26: 1.705618, 27: 1.703288, 28: 1.701131, 29: 1.699127}


def t95_one_sided(df: int) -> float:
    """One-sided 95% t quantile for ``df`` degrees of freedom (normal beyond df=29)."""
    if df < 1:
        return float("inf")
    return _T95.get(df, 1.644854)


def study_key(variant: str, comparator: str, dataset: str) -> str:
    """The registry key. Explicit triple: a study is only evidence for the pair it ran."""
    return f"{variant}|{comparator}|{dataset}"


def load_index(index_path: Optional[Path] = None) -> dict:
    """Return the (possibly empty) relabelling-study registry."""
    p = Path(index_path) if index_path else INDEX_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f).get("studies", {})


def load_study(variant: str, comparator: str, dataset: str,
               index_path: Optional[Path] = None) -> Optional[dict]:
    """Load the registered relabelling study for (variant, comparator, dataset), or None.

    D2: the study's OWN identifying fields must agree with the key it is filed under. Without
    that, a typo in ``relabel_index.json`` silently attaches the wrong study to the wrong pair —
    the registry was trusted as the sole source of truth about what a file contains.
    """
    idx = load_index(index_path)
    entry = idx.get(study_key(variant, comparator, dataset))
    if not entry:
        return None
    p = _ROOT / entry["path"]
    if not p.exists():
        return None
    with open(p) as f:
        study = json.load(f)
    claimed = (study.get("variant"), study.get("comparator"), study.get("dataset"))
    if claimed != (variant, comparator, dataset):
        study["_mismatch"] = (f"registered under {study_key(variant, comparator, dataset)!r} but "
                              f"the file itself says variant={claimed[0]!r}, "
                              f"comparator={claimed[1]!r}, dataset={claimed[2]!r}")
    study["_registry_entry"] = entry
    study["_path"] = entry["path"]
    return study


def evaluate(study: dict, min_delta_pp: float, min_relabellings: int = 4,
             variant_pct: Optional[float] = None,
             comparator_pct: Optional[float] = None,
             identity_tol_pp: float = 1e-9) -> dict:
    """Apply the rule to one relabelling study.

    Parameters
    ----------
    study
        A ``proto_P09_relabel.py`` output: ``rows`` of ``{r, identity, delta_pp, ...}``.
    min_delta_pp
        ``datasets.<ds>.min_promotion_delta_pp``. EVERY relabelling must clear it, AND the paired
        one-sided lower bound must clear it.
    min_relabellings
        Minimum number of NON-identity relabellings required for the study to count.
    variant_pct, comparator_pct
        The recorded confirm-pool means. D1: ``proto_P09_prereg.json`` makes "the identity row
        reproduces the recorded confirm numbers" a VOIDING condition, and the first draft checked
        only that an identity row EXISTED — the reproduction claim lived as English prose in
        ``relabel_index.json`` and was read by nothing. That is the same class of hole (a stated
        rule no code enforces) that P09 was convened to close. Pass them and it is enforced; omit
        them and the study is refused for want of something to check against.
    identity_tol_pp
        Tolerance for the identity comparison. These are deterministic re-runs of the same
        computation, so the honest tolerance is float-noise, not a confidence interval.

    Returns
    -------
    dict
        ``passed`` plus every number needed to state the verdict in a log entry. A study that is
        too small, or unanchored, does not pass — absence of evidence is not evidence.
    """
    rows = study.get("rows") or []
    deltas = [float(r["delta_pp"]) for r in rows]
    n_relabellings = sum(1 for r in rows if not r.get("identity"))
    identity_rows = [r for r in rows if r.get("identity")]
    failures = [dict(r=int(r["r"]), delta_pp=float(r["delta_pp"]))
                for r in rows if float(r["delta_pp"]) <= min_delta_pp]

    reasons = []
    if not rows:
        reasons.append("the study has no rows")
    if study.get("_mismatch"):
        reasons.append(f"the study does not identify itself as the one requested: "
                       f"{study['_mismatch']}")
    if n_relabellings < int(min_relabellings):
        reasons.append(f"only {n_relabellings} non-identity relabelling(s), "
                       f"{min_relabellings} required")
    for f in failures:
        reasons.append(f"relabelling r={f['r']} gives delta {f['delta_pp']:+.5f} pp, which does "
                       f"not clear the minimum effect size {min_delta_pp:+.5f} pp")

    # ── D1: the identity row must REPRODUCE the recorded confirm pool, not merely exist. ──
    identity_ok = False
    identity_detail = None
    if not identity_rows:
        reasons.append("the study has no identity (r=0) row, so it is not anchored to the "
                       "recorded confirm numbers")
    elif variant_pct is None or comparator_pct is None:
        reasons.append("no recorded confirm means were supplied, so the pre-registered voiding "
                       "condition (the identity row must reproduce them) cannot be checked")
    else:
        row = identity_rows[0]
        dv = abs(float(row["variant"]["pct"]) - float(variant_pct))
        dc = abs(float(row["comparator"]["pct"]) - float(comparator_pct))
        identity_ok = dv <= identity_tol_pp and dc <= identity_tol_pp
        identity_detail = dict(variant_study=float(row["variant"]["pct"]),
                               variant_recorded=float(variant_pct), variant_abs_diff_pp=dv,
                               comparator_study=float(row["comparator"]["pct"]),
                               comparator_recorded=float(comparator_pct),
                               comparator_abs_diff_pp=dc, tol_pp=identity_tol_pp)
        if not identity_ok:
            reasons.append(f"the identity row does not reproduce the recorded confirm numbers "
                           f"(variant differs by {dv:.3g} pp, comparator by {dc:.3g} pp, "
                           f"tolerance {identity_tol_pp:g} pp) — by pre-registration the study "
                           f"is VOID, not merely failing")

    # ── D5: the consistent companion to the min-rule. ──
    n = len(deltas)
    mean = sd = se = lower = None
    bound_ok = False
    if n >= 2:
        mean = sum(deltas) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in deltas) / (n - 1))
        se = sd / math.sqrt(n)
        lower = mean - t95_one_sided(n - 1) * se
        bound_ok = lower > min_delta_pp
        if not bound_ok:
            reasons.append(f"the paired one-sided 95% lower bound on the mean delta "
                           f"({lower:+.6f} pp) does not clear the minimum effect size "
                           f"{min_delta_pp:+.5f} pp")
    else:
        reasons.append("fewer than 2 points: no paired bound can be computed")

    passed = bool(rows) and not reasons

    return dict(
        passed=passed,
        n_points=n,
        n_relabellings=n_relabellings,
        has_identity=bool(identity_rows),
        identity_reproduces=identity_ok,
        identity_detail=identity_detail,
        min_delta_pp=float(min_delta_pp),
        min_relabellings=int(min_relabellings),
        delta_min_pp=min(deltas) if deltas else None,
        delta_max_pp=max(deltas) if deltas else None,
        delta_mean_pp=mean,
        delta_sd_pp=sd,
        delta_se_pp=se,
        paired_lower_bound_pp=lower,
        paired_bound_passed=bound_ok,
        min_rule_passed=bool(rows) and not failures,
        n_failures=len(failures),
        failures=failures,
        reasons=reasons,
        study_path=study.get("_path"),
    )
