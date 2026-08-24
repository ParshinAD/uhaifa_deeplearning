"""Probe: randomized topological reshuffle (TopoShuffle) of the forward-DAG of a
given ordering. Guaranteed score-non-decreasing; measures how much it gains and
how far it moves in permutation space."""
import numpy as np, time, sys, heapq
SP="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university-deeplearning-thesis-nn-backward-connections/a809f69f-2439-4704-95b8-1751c626882b/scratchpad"
si,ti,w = np.load(SP+"/conn_idx.npy"); si=si.astype(np.int32); ti=ti.astype(np.int32); w=w.astype(np.int64)
n=136648; m=len(si); W=int(w.sum())
pos=np.load(sys.argv[1]); r=np.empty(n,np.int32); r[np.argsort(pos,kind="stable")]=np.arange(n,dtype=np.int32)
sc=lambda r:int(w[(r[si]<r[ti])].sum())
s0=sc(r); print("start %d = %.6f%%"%(s0,100*s0/W),flush=True)

fwd = r[si]<r[ti]
fs, ft = si[fwd], ti[fwd]
import scipy.sparse as sp
A=sp.csr_matrix((np.ones(fs.size,np.int8),(fs,ft)),shape=(n,n)); A.sum_duplicates()
indptr,indices=A.indptr,A.indices
indeg0=np.zeros(n,np.int32); np.add.at(indeg0,ft,1)
# dedupe indeg to match A
indeg0=np.asarray(A.sum(axis=0)).ravel().astype(np.int32)

def toposhuffle(seed):
    rng=np.random.default_rng(seed)
    indeg=indeg0.copy()
    key=rng.random(n)
    h=[(key[v],v) for v in np.flatnonzero(indeg==0)]
    heapq.heapify(h)
    out=np.empty(n,np.int32); k=0
    while h:
        _,u=heapq.heappop(h); out[k]=u; k+=1
        for x in indices[indptr[u]:indptr[u+1]]:
            indeg[x]-=1
            if indeg[x]==0: heapq.heappush(h,(key[x],x))
    assert k==n, k
    rr=np.empty(n,np.int32); rr[out]=np.arange(n,dtype=np.int32)
    return rr

for seed in range(4):
    t=time.time(); rr=toposhuffle(seed); s=sc(rr)
    # spearman-ish displacement
    disp=np.abs(rr.astype(np.int64)-r.astype(np.int64))
    print("seed %d: score %d = %.6f%%  (delta %+d = %+.4f pp)  mean|rank shift| %.0f  max %d  time %.0fs"%(
        seed,s,100*s/W,s-s0,100*(s-s0)/W,disp.mean(),disp.max(),time.time()-t),flush=True)
