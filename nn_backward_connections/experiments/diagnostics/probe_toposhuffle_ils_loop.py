import numpy as np, time, heapq, scipy.sparse as sp
from mfas.io import load_dataset
from mfas.refine.insertion import sift
g=load_dataset("connectome"); n=g.n_nodes
si=np.asarray(g.src); ti=np.asarray(g.tgt); w=np.asarray(g.weight).astype(np.int64); W=int(w.sum())
pos=np.load("results/20260622T213942Z-H35-connectome-s42-implement-8f52fb_positions.npy")
r=np.empty(n,np.int64); r[np.argsort(pos,kind="stable")]=np.arange(n)
sc=lambda rr:int(w[(rr[si]<rr[ti])].sum())
def toposhuffle(r,seed):
    fwd=r[si]<r[ti]
    A=sp.csr_matrix((np.ones(int(fwd.sum()),np.int8),(si[fwd],ti[fwd])),shape=(n,n)); A.sum_duplicates()
    ip,ix=A.indptr,A.indices; indeg=np.asarray(A.sum(axis=0)).ravel().astype(np.int32)
    rng=np.random.default_rng(seed); key=rng.random(n)
    h=[(key[v],int(v)) for v in np.flatnonzero(indeg==0)]; heapq.heapify(h)
    out=np.empty(n,np.int64); k=0
    while h:
        _,u=heapq.heappop(h); out[k]=u; k+=1
        for x in ix[ip[u]:ip[u+1]]:
            indeg[x]-=1
            if indeg[x]==0: heapq.heappush(h,(key[x],int(x)))
    rr=np.empty(n,np.int64); rr[out]=np.arange(n); return rr
best=r.copy(); bs=sc(best); t0=time.time()
print("round  after_topo   after_sift   best      wall")
for k in range(12):
    rt=toposhuffle(best,k); st=sc(rt)
    rr,ss,_=sift(g,rt,max_sweeps=4)
    ss=int(ss)
    if ss>bs: bs=ss; best=rr.copy()
    print("%2d   %.6f%%  %.6f%%  %.6f%%  %.0fs"%(k,100*st/W,100*ss/W,100*bs/W,time.time()-t0),flush=True)
