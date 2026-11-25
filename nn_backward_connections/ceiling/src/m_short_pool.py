"""Concatenate column files (keys c<len>_<tag>, file order kept) into one pool.
Usage: m_short_pool.py <out.npz> <in1.npz> <in2.npz> ..."""
import sys

import numpy as np

arrs = {}
for p in sys.argv[2:]:
    z = np.load(p)
    for k in z.files:
        if k.startswith("c"):
            assert k not in arrs, k
            arrs[k] = z[k]
np.savez(sys.argv[1], **arrs)
print({k: v.shape for k, v in arrs.items()})
