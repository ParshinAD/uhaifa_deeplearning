import sys, numpy as np
from pathlib import Path
ROOT = Path("/Users/abed359/IdeaProjects/university/deeplearning_thesis/nn_backward_connections")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"experiments"))
from mfas import io
import size_collective_moves as S

g = io.load_dataset("mouse")
src=np.asarray(g.src,np.int64); tgt=np.asarray(g.tgt,np.int64); w=np.asarray(g.weight)
rank=S.rank_of(np.load(S.latest_h35_positions("mouse",42)))
orig = S._pair_gain_curve

def run(tag, fn, ve=1):
    S._pair_gain_curve = fn
    try:
        st = S.size_s1(g, rank, src, tgt, w, 123, [], ve)
        print(f"  {tag:<40} NO FIRE  (moves_applied={st['moves_applied']}, gain={st['realized_gain_weight']:.3g})")
    except AssertionError as e:
        print(f"  {tag:<40} FIRED    {str(e)[:60]}")
    finally:
        S._pair_gain_curve = orig

run("control (unmodified)",            orig)
run("roll(+1)",  lambda *a: np.roll(orig(*a), 1))
run("x1.5",      lambda *a: orig(*a)*1.5)
run("drop last element (argmax shift)",lambda *a: np.concatenate((orig(*a)[:-1], [0.0])) if orig(*a).size>1 else orig(*a))
run("reverse",   lambda *a: orig(*a)[::-1])
