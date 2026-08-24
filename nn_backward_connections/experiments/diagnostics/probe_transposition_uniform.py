"""Quick research probe (NOT a logged experiment): is the H35 champion ordering
2-exchange (transposition) optimal?  Delta-evaluated random-pair hill climb."""
import numpy as np, time, sys
SP="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university-deeplearning-thesis-nn-backward-connections/a809f69f-2439-4704-95b8-1751c626882b/scratchpad"
si,ti,w = np.load(SP+"/conn_idx.npy")
si=si.astype(np.int32); ti=ti.astype(np.int32); w=w.astype(np.int64)
n=136648; m=len(si); W=int(w.sum())

pos = np.load(sys.argv[1])
assert pos.shape[0]==n
r = np.empty(n,dtype=np.int32); r[np.argsort(pos,kind="stable")]=np.arange(n,dtype=np.int32)

def score(r):
    return int(w[(r[si]<r[ti])].sum())
s0=score(r); print("start score %d = %.6f%%"%(s0,100*s0/W),flush=True)

# combined incidence: for node u, entries (other, weight, sign) sign=+1 if edge u->other
nb_node = np.concatenate([si,ti]); nb_other=np.concatenate([ti,si])
nb_w = np.concatenate([w,w]); nb_out=np.concatenate([np.ones(m,np.int8),np.zeros(m,np.int8)])
o=np.argsort(nb_node,kind="stable")
nb_node=nb_node[o]; nb_other=nb_other[o]; nb_w=nb_w[o]; nb_out=nb_out[o]
ptr=np.zeros(n+1,dtype=np.int64); np.add.at(ptr,nb_node+1,1); ptr=np.cumsum(ptr)
deg=(ptr[1:]-ptr[:-1])

def delta(u,v):
    """swap ranks of u and v; returns exact int64 gain."""
    a,b=int(r[u]),int(r[v])
    if a>b: u,v=v,u; a,b=b,a
    g=0
    for node,sgn in ((u,1),(v,-1)):
        s,e=ptr[node],ptr[node+1]
        oth=nb_other[s:e]; ww=nb_w[s:e]; isout=nb_out[s:e]
        ro=r[oth]
        msk=(ro>a)&(ro<b)
        if msk.any():
            # node u moving right: out-edges lost, in-edges gained  (sgn=+1)
            # node v moving left : out-edges gained, in-edges lost   (sgn=-1)
            oo=isout[msk].astype(np.int64); wm=ww[msk]
            g += int((wm*np.where(oo==1,-sgn,sgn)).sum())
    # direct u<->v edges
    s,e=ptr[u],ptr[u+1]
    oth=nb_other[s:e]
    hit=np.flatnonzero(oth==v)
    for h in hit:
        if nb_out[s+h]==1: g-=int(nb_w[s+h])   # u->v was forward, becomes backward
        else:              g+=int(nb_w[s+h])   # v->u was backward, becomes forward
    return g

# validate delta against full rescore
rng=np.random.default_rng(1)
for _ in range(5):
    u,v=rng.integers(0,n,2)
    if u==v: continue
    d=delta(u,v); r2=r.copy(); r2[u],r2[v]=r[v],r[u]
    assert score(r2)-s0==d, (u,v,d,score(r2)-s0)
print("delta validated",flush=True)

t=time.time(); acc=0; tried=0; gain=0
BUDGET=float(sys.argv[2]) if len(sys.argv)>2 else 420.0
B=20000
while time.time()-t<BUDGET:
    U=rng.integers(0,n,B); V=rng.integers(0,n,B)
    for k in range(B):
        u,v=int(U[k]),int(V[k])
        if u==v: continue
        tried+=1
        d=delta(u,v)
        if d>0:
            r[u],r[v]=r[v],r[u]; gain+=d; acc+=1
    if time.time()-t>BUDGET: break
print("proposals %d  accepted %d  gain %d (= %.4f pp)  rate %.0f/s  time %.0fs"%(
    tried,acc,gain,100*gain/W,tried/(time.time()-t),time.time()-t),flush=True)
print("final score %d = %.6f%%"%(s0+gain,100*(s0+gain)/W))
