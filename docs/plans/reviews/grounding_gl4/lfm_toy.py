# Toy check for the GL4-GL5 confounding lens, D1 (T8 (b) design review, 2026-10-24).
# Copied verbatim from the reviewer's scratch file; it is a fixed-kernel ridge on a scripted A/B/C/D
# triad, NOT the composed loop (the composed-loop toy owed by G21 is a separate artefact).
# Run: python docs/plans/reviews/grounding_gl4/lfm_toy.py
import numpy as np
rng=np.random.default_rng(0)
def cos(a,b): return a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
obj={'A':np.array([0.9,0.1,0.2]),'C':np.array([0.85,0.15,0.25]),'B':np.array([0.1,0.9,0.2]),'D':np.array([0.2,0.2,0.9])}
O={'O1':np.array([1.0,0,0]),'O2':np.array([0,1.0,0]),'O3':np.array([0,0,1.0]),'O4':np.array([0.5,0.5,0])}
ident={k:rng.normal(size=8) for k in obj}  # neutral names
for k in ident: ident[k]=ident[k]*0.3+np.ones(8)  # all high cosine, like shared affordance word
def ridge(X,Y,lam=0.1,w=(1,1,1)):
    n=len(X); K=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j]=kern(X[i],X[j],w)
    a=np.linalg.solve(K+lam*np.eye(n),Y)
    return lambda x: np.array([kern(x,X[j],w) for j in range(n)])@a
def kern(x,y,w):
    return (w[0]*cos(x[0],y[0])+w[1]*cos(x[1],y[1])+w[2]*cos(x[2],y[2]))/sum(w)
def run(sched_situ, w=(1,1,1),lam=0.1):
    lab1={'A':'O1','B':'O1','C':'O2','D':'O4'}
    X=[];Y=[]
    for it,sit in sched_situ: X.append((ident[it],obj[it],sit)); Y.append(O[lab1[it]])
    f1=ridge(X,np.array(Y),lam,w)
    Y2=[O['O3'] if it=='A' else O[lab1[it]] for it,_ in sched_situ]
    f2=ridge(X,np.array(Y2),lam,w)
    probe_sit=np.array([0.5,0.5,0.5,0.5])
    out={}
    for it in 'BC':
        x=(ident[it],obj[it],probe_sit)
        out[it]=(f2(x)-f1(x))[2]  # inheritance of O3 dimension
    return out
neutral=np.array([0.5,0.5,0.5,0.5])
s1=[(it,neutral+rng.normal(scale=0.02,size=4)) for it in 'ABCD' for _ in range(5)]
print('case1 neutral situation:',run(s1))
# case2: blocked schedule; situation reflects previous consequence (after-O1 state vs after-O2)
after={'O1':np.array([0.9,0.1,0.5,0.5]),'O2':np.array([0.1,0.9,0.5,0.5]),'O4':np.array([0.5,0.5,0.9,0.1])}
lab1={'A':'O1','B':'O1','C':'O2','D':'O4'}
s2=[]
for it in 'ABCD':
    for t in range(5):
        prev = lab1[it]  # previous trial was same item (blocked)
        s2.append((it,after[prev]+rng.normal(scale=0.02,size=4)))
print('case2 blocked, situation carries prior consequence:',run(s2))
print('case2, situation block weight 3:',run(s2,w=(1,1,3)))
# sensed-similarity no-learning baseline: delta spread by object cosine
for it in 'BC': print('sensed baseline cos to A',it,cos(obj[it],obj['A']))
def run_probe(sched, probe_sits, w=(1,1,1), lam=0.1):
    X=[];Y=[];Y2=[]
    for it,sit in sched:
        X.append((ident[it],obj[it],sit)); Y.append(O[lab1[it]]); Y2.append(O['O3'] if it=='A' else O[lab1[it]])
    f1=ridge(X,np.array(Y),lam,w); f2=ridge(X,np.array(Y2),lam,w)
    return {it:round(float((f2((ident[it],obj[it],probe_sits[it]))-f1((ident[it],obj[it],probe_sits[it])))[2]),3) for it in 'BC'}
print('probe in each item own habitual state:',run_probe(s2,{'B':after['O1'],'C':after['O2']}))
print('probe both in after-O1 state:',run_probe(s2,{'B':after['O1'],'C':after['O1']}))
print('same, situation weight 3, lam 1:',run_probe(s2,{'B':after['O1'],'C':after['O2']},w=(1,1,3),lam=1.0))
print('same, situation weight 6, lam 1:',run_probe(s2,{'B':after['O1'],'C':after['O2']},w=(1,1,6),lam=1.0))
