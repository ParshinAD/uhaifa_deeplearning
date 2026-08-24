"""Back-edge-targeted transposition probe (contrast with uniform-pair swaps)."""
import numpy as np, time, sys
SP="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university-deeplearning-thesis-nn-backward-connections/a809f69f-2439-4704-95b8-1751c626882b/scratchpad"
si,ti,w=np.load(SP+"/conn_idx.npy"); si=si.astype(np.int32); ti=ti.astype(np.int32); w=w.astype(np.int64)
n=136648; m=len(si); W=int(w.sum())
pos=np.load(sys.argv[1]); r=np.empty(n,np.int32); r[np.argsort(pos,kind="stable")]=np.arange(n,dtype=np.int32)
sc=lambda r:int(w[(r[si]<r[ti])].sum()); s0=sc(r); print("start %.6f%%"%(100*s0/W),flush=True)
nb_node=np.concatenate([si,ti]); nb_other=np.concatenate([ti,si]); nb_w=np.concatenate([w,w])
nb_out=np.concatenate([np.ones(m,np.int8),np.zeros(m,np.int8)])
o=np.argsort(nb_node,kind="stable"); nb_node=nb_node[o]; nb_other=nb_other[o]; nb_w=nb_w[o]; nb_out=nb_out[o]
ptr=np.zeros(n+1,np.int64); np.add.at(ptr,nb_node+1,1); ptr=np.cumsum(ptr)
def delta(u,v):
    a,b=int(r[u]),int(r[v])
    if a>b: u,v=v,u; a,b=b,a
    g=0
    for node,sgn in ((u,1),(v,-1)):
        s,e=ptr[node],ptr[node+1]; oth=nb_other[s:e]; ww=nb_w[s:e]; io=nb_out[s:e]; ro=r[oth]
        msk=(ro>a)&(ro<b)
        if msk.any():
            oo=io[msk].astype(np.int64); g+=int((ww[msk]*np.where(oo==1,-sgn,sgn)).sum())
    s,e=ptr[u],ptr[u+1]; oth=nb_other[s:e]
    for h in np.flatnonzero(oth==v):
        g += -int(nb_w[s+h]) if nb_out[s+h]==1 else int(nb_w[s+h])
    return g
rng=np.random.default_rng(7); t=time.time(); gain=0; acc=0; tried=0
BUD=float(sys.argv[2])
while time.time()-t<BUD:
    back=np.flatnonzero(r[si]>r[ti])
    p=w[back].astype(np.float64); p/=p.sum()
    pick=rng.choice(back,size=200000,p=p)
    for e in pick:
        tried+=1
        u,v=int(si[e]),int(ti[e])
        if r[u]<=r[v]: continue
        d=delta(u,v)
        if d>0: r[u],r[v]=r[v],r[u]; gain+=d; acc+=1
        if time.time()-t>BUD: break
print("proposals %d accepted %d gain %d (%+.4f pp) rate %.0f/s"%(tried,acc,gain,100*gain/W,tried/(time.time()-t)))
print("final %.6f%%"%(100*(s0+gain)/W))
