#!/usr/bin/env python
"""Seed-dependence classifier — decides how many screen seeds a variant actually needs.

Campaign context (queue item P02, 2026-08-09)
---------------------------------------------
On this machine the champion pipelines are DETERMINISTIC: they accept a ``seed`` but never
draw from it, because ``init_positions`` comes from deterministic greedy-FAS and
``make_init_positions`` — the only RNG consumer in ``run_rocket`` — is therefore never
reached. Running such a variant at seeds 42/123/999 executes the same computation three
times. On microns (3240 s/run) that is 108 minutes of screen wall-clock buying nothing.

This script decides, MECHANICALLY, whether a variant draws from its seed, so the screen can
run 1 seed for the ones that do not. It exists because a hand-maintained table is exactly the
kind of thing that rots: the first hand classification of the 16 current variants already got
H19 wrong (it looks stochastic, but ``_init_positions`` takes no seed).

Why a call graph and not a grep
-------------------------------
``mfas/refine/insertion.py`` contains BOTH ``sift`` (no RNG — used by H30) and
``sift_gauss_seidel_ref`` (``RandomState(seed)``, an unused cross-check kernel). A file-level
scan flags that module and therefore misclassifies H30 and H35, which is every champion we
have. Classification must follow which FUNCTIONS are actually reachable from ``run()``.

The safety asymmetry
--------------------
A false "rng" costs wall-clock. A false "deterministic" silently discards real variance and
corrupts a verdict. So every ambiguity resolves to ``rng``:

* a call this script cannot resolve to a first-party function is reported as UNRESOLVED and
  forces ``rng`` (``--strict``, the default);
* ``run_rocket`` called without an explicit ``init_positions=`` argument is ``rng``
  (it falls through to ``torch.randn`` / ``make_init_positions``);
* a static verdict of ``deterministic`` is a NECESSARY condition for the 1-seed screen, never
  a sufficient one — ``autoresearch/CAMPAIGN.md`` also requires the cheap mouse probe to
  corroborate it (that probe can escalate to 3 seeds, never de-escalate).

The converse trap, which is why the static answer overrides the empirical one: H31 DOES draw
from its seed (``RandomState(seed + 7919)`` in the LNS destroy operator), but on mouse its LNS
stage never beats the sift, so best-by-oracle returns the deterministic sift vector and the
variant LOOKS seed-inert. Observed determinism on a small proxy is not determinism.

Usage
-----
    python autoresearch/seed_class.py --variant H35
    python autoresearch/seed_class.py --all [--out autoresearch/seed_class.json]
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
_PKG = _SRC / "mfas"

# ── RNG draw sites ────────────────────────────────────────────────────────────
# Attribute calls whose LAST component means "draw a random number". Seeding calls
# (np.random.seed, torch.manual_seed, random.seed) are deliberately NOT here: seeding without
# drawing is what the champion pipelines do, and it is precisely what makes them deterministic.
RNG_ATTRS = {
    "randn", "rand", "randint", "randperm", "random_sample", "random_integers",
    "choice", "shuffle", "permutation", "uniform", "normal", "standard_normal",
    "binomial", "poisson", "exponential", "beta", "gamma", "sample", "randrange",
    "rand_like", "randn_like", "randint_like", "multinomial", "bernoulli",
}
# Names that CONSTRUCT a random generator. Constructing one from `seed` is a draw for our
# purposes: the object's whole point is to be consumed.
RNG_CTORS = {"RandomState", "default_rng", "Generator", "Random", "SeedSequence"}

# Reading the clock and branching on it is non-determinism that is not RNG: a stage with a
# time budget does fewer passes on a loaded machine and returns a different order. Never
# treat these as safe, however they are imported.
_CLOCK_ROOTS = {"time", "timeit", "datetime"}

_BUILTINS = frozenset(dir(builtins))

# Third-party / stdlib roots we do not descend into but consider RNG-free unless the call
# itself matches RNG_ATTRS / RNG_CTORS above.
KNOWN_SAFE_ROOTS = {
    "np", "numpy", "torch", "pd", "pandas", "scipy", "sp", "math", "time", "json",
    "os", "sys", "collections", "itertools", "functools", "dataclasses", "typing",
    "logging", "pathlib", "warnings", "heapq", "copy", "re", "hashlib", "F", "optim", "nn",
}


class ModuleIndex:
    """Parsed first-party modules + the import bindings needed to resolve calls."""

    def __init__(self) -> None:
        self.trees: Dict[str, ast.Module] = {}          # dotted mfas module -> AST
        self.funcs: Dict[Tuple[str, str], ast.AST] = {}  # (module, qualname) -> FunctionDef
        # module -> {local name: (target_module, target_name)}; target_name None = module alias
        self.binds: Dict[str, Dict[str, Tuple[str, Optional[str]]]] = {}
        # module -> module-level assigned names (``_EPOCHS``, ``_ALPHA``, ``ID`` …). Method
        # calls on these (``_EPOCHS.get(...)``) are dict/constant access, not function calls.
        self.globals: Dict[str, Set[str]] = {}
        self._load()

    # ---- loading -------------------------------------------------------------
    def _mod_name(self, path: Path) -> str:
        rel = path.relative_to(_SRC).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _load(self) -> None:
        for path in sorted(_PKG.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            mod = self._mod_name(path)
            self.trees[mod] = tree
            self.binds[mod] = {}
            self._index_funcs(mod, tree, prefix="")
            self._index_imports(mod, tree)
            gl: Set[str] = set()
            for stmt in tree.body:
                targets = (stmt.targets if isinstance(stmt, ast.Assign)
                           else [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
                for t in targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Name):
                            gl.add(sub.id)
            self.globals[mod] = gl

    def _index_funcs(self, mod: str, node: ast.AST, prefix: str) -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                self.funcs[(mod, name)] = child
                # Nested defs are reachable only through the parent; index them flat too.
                self._index_funcs(mod, child, prefix=name + ".")
            elif isinstance(child, ast.ClassDef):
                self._index_funcs(mod, child, prefix=prefix + child.name + ".")

    def _resolve_relative(self, mod: str, level: int, module: Optional[str]) -> str:
        """Turn a relative import inside ``mod`` into an absolute mfas module path."""
        base = mod.split(".")
        # level 1 = current package. For a module (not a package __init__) drop its own name.
        if (_SRC / Path(*base) / "__init__.py").exists():
            pkg = base                      # mod is a package
        else:
            pkg = base[:-1]                 # mod is a module inside a package
        if level > 1:
            pkg = pkg[: len(pkg) - (level - 1)]
        return ".".join(pkg + ([module] if module else []))

    def _index_imports(self, mod: str, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.binds[mod][a.asname or a.name.split(".")[0]] = (a.name, None)
            elif isinstance(node, ast.ImportFrom):
                target = (self._resolve_relative(mod, node.level, node.module)
                          if node.level else (node.module or ""))
                for a in node.names:
                    self.binds[mod][a.asname or a.name] = (target, a.name)

    # ---- resolution ----------------------------------------------------------
    def resolve_in(self, mod: str, enclosing: str, name: str) -> Optional[Tuple[str, str]]:
        """Resolve ``name`` called from inside function ``enclosing`` of ``mod``.

        Closures come first: ``run_rocket`` calls its own nested ``discrete_score``, and
        ``greedy_fas_order`` its own nested ``remove``. Both are indexed under the dotted
        qualname of their parent, so a flat lookup misses them and — fail-safe — that would
        report them UNRESOLVED and force ``rng`` on every champion.
        """
        parts = enclosing.split(".") if enclosing else []
        while parts:
            cand = ".".join(parts + [name])
            if (mod, cand) in self.funcs:
                return (mod, cand)
            parts.pop()
        return self.resolve(mod, name)

    def resolve(self, mod: str, name: str) -> Optional[Tuple[str, str]]:
        """Resolve a bare call name in ``mod`` to a first-party (module, func), following
        package re-exports such as ``from .insertion import sift`` in ``refine/__init__``."""
        if (mod, name) in self.funcs:
            return (mod, name)
        bound = self.binds.get(mod, {}).get(name)
        if not bound:
            return None
        tmod, tname = bound
        if tname is None or not tmod.startswith("mfas"):
            return None
        if (tmod, tname) in self.funcs:
            return (tmod, tname)
        # Re-export: the target module imported it from somewhere else.
        if tmod in self.binds:
            hop = self.binds[tmod].get(tname)
            if hop and hop[1] and hop[0].startswith("mfas") and (hop[0], hop[1]) in self.funcs:
                return (hop[0], hop[1])
        return None


def _attr_chain(node: ast.AST) -> Tuple[List[str], Optional[str]]:
    """(['np','random','randn'], 'np') for ``np.random.randn``.

    The second element is the chain's ROOT NAME, or ``None`` when the chain bottoms out on a
    computed value — ``(w_np / max_w).astype(...)``, ``f(x).sum()``. Those are methods on a
    temporary, never a first-party function this script could resolve, so they must not be
    reported UNRESOLVED (under the fail-safe rule that would force ``rng`` on every variant).
    """
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    root = cur.id if isinstance(cur, ast.Name) else None
    if root is not None:
        parts.append(root)
    return list(reversed(parts)), root


def _local_names(fn: ast.AST) -> Set[str]:
    """Parameters and assigned names inside ``fn``.

    A call like ``order.astype(...)`` or ``seq.remove(...)`` is a METHOD on a local object,
    not a first-party function this script could resolve. Without this set every such call
    would be reported UNRESOLVED and — under the fail-safe rule — force ``rng``, which is how
    the first version misclassified H16 and H19.
    """
    names: Set[str] = set()
    # Parameters of THIS function and of every function nested inside it. ast.walk descends
    # into nested defs, so their calls are scanned here too and their parameters
    # (``discrete_score(pos_param)``) must be in scope or they read as unresolved globals.
    for node in ast.walk(fn):
        a = getattr(node, "args", None)
        if a is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                              ast.Lambda)):
            continue
        for grp in (a.args, a.posonlyargs, a.kwonlyargs):
            names.update(x.arg for x in grp or [])
        for x in (a.vararg, a.kwarg):
            if x:
                names.add(x.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            tgt = node.target
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for sub in ast.walk(node.optional_vars):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def _prune_dead_init_branches(fn: ast.AST, passed_not_none: Set[str]) -> Set[int]:
    """Line numbers made unreachable by arguments the caller passed as non-None.

    ``run_rocket(g, cfg, seed, device, init_positions=<tensor>)`` cannot reach its own
    ``elif cfg.init_mode == 'random': torch.randn(n)`` / ``else: make_init_positions(...)``
    arms — the first branch of ``if init_positions is not None:`` wins. Without modelling
    that, every champion pipeline is classified ``rng`` through a branch it never executes,
    which is exactly the misclassification this whole script exists to prevent.

    The rule is deliberately narrow and syntactic: for ``if <p> is not None`` where ``p`` was
    passed non-None, the else-arm is dead; for ``if <p> is None``, the then-arm is dead.
    Anything more clever would be a dataflow analysis, and a wrong one would be worse than none.
    """
    dead: Set[int] = set()
    if not passed_not_none:
        return dead
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.left, ast.Name)
                and test.left.id in passed_not_none
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None):
            continue
        if isinstance(test.ops[0], ast.IsNot):        # `p is not None` -> else-arm is dead
            doomed = node.orelse
        elif isinstance(test.ops[0], ast.Is):         # `p is None`     -> then-arm is dead
            doomed = node.body
        else:
            continue
        for stmt in doomed:
            for sub in ast.walk(stmt):
                if hasattr(sub, "lineno"):
                    dead.add(sub.lineno)
    return dead


def _dataset_constants(tree: ast.Module, dataset: str) -> Dict[str, object]:
    """Module-level ``NAME = {"<dataset>": <constant>, ...}`` tables, resolved for ``dataset``.

    Deliberately literal-only: every key must be a string constant and every value a
    constant. A table built at runtime resolves to nothing and therefore prunes nothing,
    which is the fail-safe direction.
    """
    out: Dict[str, object] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Name) and isinstance(stmt.value, ast.Dict)):
            continue
        table = {}
        ok = True
        for k, v in zip(stmt.value.keys, stmt.value.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)
                    and isinstance(v, ast.Constant)):
                ok = False
                break
            table[k.value] = v.value
        if ok and dataset in table:
            out[tgt.id] = table[dataset]
    return out


def _prune_dataset_gated_branches(fn: ast.AST, consts: Dict[str, object]) -> Set[int]:
    """Line numbers unreachable because a per-dataset constant switches the stage OFF.

    The concrete case this exists for, and the only shape it recognises::

        max_pops = _PAIR_MAX_POPS.get(g.name, 0)     # {"connectome": 0, "microns": 0, ...}
        if max_pops:
            pair_relocate(...)                        # reads _time.time() -> UNRESOLVED -> rng

    On connectome and microns that body is dead, so the variant is genuinely deterministic
    there, yet the classifier called it ``rng`` and the screen paid for three bit-identical
    seeds. H64's three connectome screen runs agreed to every digit, which is the symptom.

    Narrow and syntactic on purpose, exactly like _prune_dead_init_branches: the local must
    be assigned from ``TABLE.get(g.name, <const>)`` or ``TABLE[g.name]`` with TABLE resolved
    literally, and the test must be a bare truthiness check on that local. Anything else
    prunes nothing. Pruning too little costs wall-clock; pruning too much would discard real
    variance, so every ambiguity resolves to no-prune.
    """
    dead: Set[int] = set()
    if not consts:
        return dead

    bound: Dict[str, object] = {}
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        val = node.value
        table = None
        if (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute)
                and val.func.attr == "get" and isinstance(val.func.value, ast.Name)):
            table = val.func.value.id
        elif isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name):
            table = val.value.id
        if table in consts:
            bound[node.targets[0].id] = consts[table]

    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Name) and test.id in bound:
            doomed = node.body if not bound[test.id] else node.orelse
        elif (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
              and isinstance(test.operand, ast.Name) and test.operand.id in bound):
            doomed = node.body if bound[test.operand.id] else node.orelse
        else:
            continue
        for stmt in doomed:
            for sub in ast.walk(stmt):
                if hasattr(sub, "lineno"):
                    dead.add(sub.lineno)
    return dead


def _local_thirdparty_imports(fn: ast.AST) -> Set[str]:
    """Names bound by a FUNCTION-LOCAL import of a third-party module.

    ``mfas/refine/reclaim2.py`` does ``from scipy.sparse.csgraph import connected_components``
    INSIDE resolve_conflicts_scc. _index_imports only records MODULE-level imports, so that
    name resolved to nothing, was reported UNRESOLVED, and forced ``rng`` on what is a pure
    graph algorithm -- which is why H63 and H64 screened at 3 bit-identical seeds.

    Only imports rooted in KNOWN_SAFE_ROOTS count, and the RNG_CTORS / RNG_ATTRS checks still
    run BEFORE this one, so ``from numpy.random import default_rng`` is still caught as RNG
    rather than waved through here. Pruning too little costs wall-clock; pruning too much
    would discard real variance, so anything unrecognised stays unresolved.

    CLOCK MODULES ARE EXCLUDED, and that exclusion is the point rather than an oversight.
    ``mfas/refine/pair_relocate.py:178`` does ``import time as _time`` inside the function and
    then breaks its own loop on ``_time.time() - t0 > time_budget_s``, so the result depends
    on how loaded the box is -- non-determinism that is not RNG. Today the classifier catches
    it only by accident, because the local alias fails to resolve; whitelisting ``time`` here
    would have silenced that accident and called a clock-truncated stage deterministic. The
    microns run truncated on 2026-08-26 is what that costs.
    """
    names: Set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in KNOWN_SAFE_ROOTS and root not in _CLOCK_ROOTS:
                for a in node.names:
                    names.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in KNOWN_SAFE_ROOTS and root not in _CLOCK_ROOTS:
                    names.add(a.asname or a.name.split(".")[0])
    return names


def _definitely_not_none(fn: ast.AST) -> Set[str]:
    """Local names in ``fn`` that provably hold a non-None value where they are used.

    Used to decide whether passing ``x`` to a callee's optional parameter licenses pruning
    that callee's ``is None`` branch. Deliberately conservative: a parameter whose default is
    ``None`` (``time_limit=None``) never qualifies, so we never prune a branch that a runtime
    ``None`` would actually take.
    """
    none_params: Set[str] = set()
    a = getattr(fn, "args", None)
    if a is not None:
        pos = list(a.posonlyargs or []) + list(a.args or [])
        for arg, dflt in zip(pos[len(pos) - len(a.defaults or []):], a.defaults or []):
            if isinstance(dflt, ast.Constant) and dflt.value is None:
                none_params.add(arg.arg)
        for arg, dflt in zip(a.kwonlyargs or [], a.kw_defaults or []):
            if isinstance(dflt, ast.Constant) and dflt.value is None:
                none_params.add(arg.arg)

    ok: Set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        if isinstance(val, ast.Constant) and val.value is None:
            continue
        # `x = None if c else y` can be None — do not trust it.
        if isinstance(val, ast.IfExp) and any(
                isinstance(b, ast.Constant) and b.value is None for b in (val.body, val.orelse)):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                ok.add(tgt.id)
    return ok - none_params


def analyze(idx: ModuleIndex, variant: str, strict: bool = True,
            dataset: Optional[str] = None) -> dict:
    """Classify one variant by walking every function reachable from its ``run()``.

    Walk state is (module, function, params-known-non-None-at-this-call-site). The third
    element is what lets ``run_rocket`` be analyzed correctly in both of its uses: reached
    with ``init_positions`` bound it is RNG-free, reached without it, it draws.
    """
    entry_mod = f"mfas.experiments.{variant}"
    if (entry_mod, "run") not in idx.funcs:
        return {"variant": variant, "verdict": "error",
                "reason": f"no run() found in {entry_mod}"}

    rng_hits: List[str] = []
    unresolved: List[str] = []
    visited: Set[Tuple[str, str, frozenset]] = set()
    stack: List[Tuple[str, str, frozenset]] = [(entry_mod, "run", frozenset())]

    while stack:
        mod, fn, bound = stack.pop()
        if (mod, fn, bound) in visited:
            continue
        visited.add((mod, fn, bound))
        node = idx.funcs.get((mod, fn))
        if node is None:
            continue

        dead = _prune_dead_init_branches(node, set(bound))
        if dataset is not None:
            dead |= _prune_dataset_gated_branches(
                node, _dataset_constants(idx.trees.get(mod, ast.Module(body=[],
                                                                      type_ignores=[])),
                                         dataset))
        # Closure scope: a nested function sees its enclosing functions' locals
        # (``remove()`` inside ``greedy_fas_order`` calls ``sinks.append``, and ``sinks``
        # belongs to the parent). Walk the qualname chain outward.
        locals_ = _local_names(node)
        parts = fn.split(".")[:-1]
        while parts:
            outer = idx.funcs.get((mod, ".".join(parts)))
            if outer is not None:
                locals_ |= _local_names(outer)
            parts.pop()
        notnone = _definitely_not_none(node)
        # A third-party name imported inside this function is bound here, not at module
        # level, so fold it into the local scope before the unresolved check.
        locals_ |= _local_thirdparty_imports(node)

        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call) or sub.lineno in dead:
                continue
            func = sub.func

            if isinstance(func, ast.Attribute):
                chain, root = _attr_chain(func)
                last = chain[-1] if chain else ""
                # No root filter: any generator object was built by an RNG_CTOR we already
                # catch, so this is belt-and-braces and its false positives are the safe kind.
                if last in RNG_ATTRS or last in RNG_CTORS:
                    rng_hits.append(f"{mod}.{fn}: {'.'.join(chain)}() line {sub.lineno}")
                elif (root is None or root in KNOWN_SAFE_ROOTS or root in locals_
                      or root in idx.globals.get(mod, set())):
                    pass       # temporary, third-party, local object, or module constant
                elif idx.resolve_in(mod, fn, root) is None:
                    unresolved.append(f"{mod}.{fn}: {'.'.join(chain)}() line {sub.lineno}")

            elif isinstance(func, ast.Name):
                name = func.id
                if name in RNG_CTORS:
                    rng_hits.append(f"{mod}.{fn}: {name}() line {sub.lineno}")
                    continue
                tgt = idx.resolve_in(mod, fn, name)
                if tgt is None:
                    # `dir(__builtins__)` is NOT usable here: __builtins__ is the builtins
                    # MODULE in a __main__ script but its __dict__ when imported, so the same
                    # code classified differently from the CLI and from pytest.
                    if name not in _BUILTINS and name not in locals_ \
                            and name not in KNOWN_SAFE_ROOTS and name[0].islower() \
                            and name not in idx.globals.get(mod, set()):
                        unresolved.append(f"{mod}.{fn}: {name}() line {sub.lineno}")
                    continue
                # Which of the callee's parameters does this call site bind to a non-None value?
                callee = idx.funcs.get(tgt)
                passed: Set[str] = set()
                if callee is not None:
                    ca = getattr(callee, "args", None)
                    params = ([x.arg for x in (ca.posonlyargs or [])]
                              + [x.arg for x in (ca.args or [])]) if ca else []

                    def _is_notnone(expr: ast.AST) -> bool:
                        if isinstance(expr, ast.Constant):
                            return expr.value is not None
                        if isinstance(expr, ast.Call):
                            return True
                        if isinstance(expr, ast.Name):
                            return expr.id in notnone
                        return False

                    for i, arg in enumerate(sub.args):
                        if i < len(params) and _is_notnone(arg):
                            passed.add(params[i])
                    for kw in sub.keywords:
                        if kw.arg and _is_notnone(kw.value):
                            passed.add(kw.arg)
                stack.append((tgt[0], tgt[1], frozenset(passed)))

    if rng_hits:
        verdict = "rng"
    elif unresolved and strict:
        verdict = "rng"          # fail-safe: unknown code path -> assume it draws
    else:
        verdict = "deterministic"

    return {
        "variant": variant,
        "verdict": verdict,
        "screen_seeds_required": 1 if verdict == "deterministic" else 3,
        "rng_evidence": sorted(set(rng_hits)),
        "unresolved_calls": sorted(set(unresolved)),
        "functions_walked": len({(m, f) for m, f, _ in visited}),
        "reachable": sorted({f"{m}.{f}" for m, f, _ in visited}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", help="variant id, e.g. H35")
    ap.add_argument("--all", action="store_true", help="classify every variant module")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--lenient", action="store_true",
                    help="do NOT force 'rng' on unresolved calls (diagnostic only)")
    args = ap.parse_args()

    if not args.variant and not args.all:
        ap.error("pass --variant <id> or --all")

    idx = ModuleIndex()
    variants = ([p.stem for p in sorted((_PKG / "experiments").glob("*.py"))
                 if p.stem != "__init__"] if args.all else [args.variant])

    results = {v: analyze(idx, v, strict=not args.lenient) for v in variants}

    print(f"{'variant':22s} {'verdict':14s} {'seeds':5s}  evidence")
    for v, r in results.items():
        if r["verdict"] == "error":
            print(f"{v:22s} {'ERROR':14s} {'-':5s}  {r['reason']}")
            continue
        ev = (r["rng_evidence"][0] if r["rng_evidence"]
              else (f"UNRESOLVED: {r['unresolved_calls'][0]}" if r["unresolved_calls"] else "-"))
        print(f"{v:22s} {r['verdict']:14s} {r['screen_seeds_required']:<5d}  {ev}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"_doc": "Mechanical seed-dependence classification (queue item P02). "
                     "verdict='deterministic' is a NECESSARY condition for the 1-seed screen; "
                     "CAMPAIGN.md also requires the cheap mouse probe to corroborate it. "
                     "Regenerate: python autoresearch/seed_class.py --all --out <this file>",
             "variants": results}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
