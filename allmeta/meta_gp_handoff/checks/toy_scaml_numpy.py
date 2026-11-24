"""Quick numpy check of ScaML-GP correction speed for dissimilar targets (not for the repo)."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
rng = np.random.default_rng(0)
se = lambda A, B, l, o: o * np.exp(-0.5 * ((A[:, None, :] - B[None]) ** 2).sum(-1) / l**2)
d, M, n_src, s2 = 2, 8, 60, 0.05**2
cs = rng.uniform(0, 1, (M, 2)); amps = rng.uniform(0.7, 1.3, M)
src = [lambda X, c=c, a=a: a * np.sin(3 * X.sum(1) + 5 * c[0]) + c[1] * X[:, 0] for c, a in zip(cs, amps)]
post = []
for f in src:  # source GPs, fixed hyperparameters stand in for fitted theta_m
    Xm = rng.uniform(0, 1, (n_src, d)); ym = f(Xm) + rng.normal(0, 0.05, n_src)
    L = np.linalg.cholesky(se(Xm, Xm, 0.3, 1.0) + s2 * np.eye(n_src))
    post.append((Xm, np.linalg.solve(L.T, np.linalg.solve(L, ym)), L))
def cache(A, B):  # mu_m(A) (M,a), Sigma_m(A,B) (M,a,b): latent posteriors, Eq. 6 ingredients
    mus, sig = [], []
    for Xm, alpha, L in post:
        ka, kb = se(A, Xm, 0.3, 1.0), se(B, Xm, 0.3, 1.0)
        mus.append(ka @ alpha); sig.append(se(A, B, 0.3, 1.0) - np.linalg.solve(L, ka.T).T @ np.linalg.solve(L, kb.T))
    return np.array(mus), np.array(sig)
Xq = rng.uniform(0, 1, (300, d))
sob = qmc.Sobol(d, seed=42).random(32)
targets = {
    "similar (0.7 f1)": lambda X: 0.7 * src[0](X) + 0.1 * np.cos(4 * X[:, 1]),
    "average of sources": lambda X: np.mean([f(X) for f in src], 0),
    "unrelated": lambda X: 2 * np.cos(6 * X[:, 0]) * X[:, 1],
    "anti (-f1)": lambda X: -src[0](X),
}
mq_all, _ = cache(Xq, Xq[:1])
def parts(z, Xt, yt, C, meta):
    w = np.exp(z[:M]) if meta else np.zeros(M); lt, ot = np.exp(z[M:])
    mut, St = C
    mt = w @ mut
    Kt = se(Xt, Xt, lt, ot) + np.einsum('m,mij->ij', w**2, St) + s2 * np.eye(len(Xt))
    return w, lt, ot, mt, Kt
def nlpost(z, Xt, yt, C, meta):
    w, lt, ot, mt, Kt = parts(z, Xt, yt, C, meta)
    L = np.linalg.cholesky(Kt + 1e-9 * np.eye(len(Xt))); r = np.linalg.solve(L, yt - mt)
    lp = (-w.sum() + z[:M].sum()) if meta else 0.0           # Gamma(1,1) on w (+ log Jacobian)
    lp += -0.5 * ((z[M] - 0.5) / 1.5) ** 2 - 0.5 * ((z[M + 1] + 2) / 3) ** 2  # paper's k_t priors
    return 0.5 * r @ r + np.log(np.diag(L)).sum() - lp
print(f"{'target':20s} {'k':>3s} {'ScaML':>7s} {'single':>7s}  fitted w_m > 0.1")
for name, g in targets.items():
    yq = g(Xq)
    for k in [0, 3, 5, 10, 20, 32]:
        if k == 0:
            print(f"{name:20s} {k:3d} {np.sqrt(np.mean((mq_all.mean(0) - yq)**2)):7.3f} {np.sqrt(np.mean(yq**2)):7.3f}  w = 1/M (prior mean = average)")
            continue
        Xt = sob[:k]; yt = g(Xt) + rng.normal(0, 0.05, k)
        C = cache(Xt, Xt); mq, Sqt = cache(Xq, Xt)
        out = {}
        for meta in (True, False):
            fits = [minimize(nlpost, np.r_[np.full(M, np.log(w0)), np.log(l0), -1.0], args=(Xt, yt, C, meta), method="L-BFGS-B")
                    for w0, l0 in ((1 / M, 0.3), (0.5, 1.0))]
            z = min(fits, key=lambda r: r.fun).x
            w, lt, ot, mt, Kt = parts(z, Xt, yt, C, meta)
            kqt = se(Xq, Xt, lt, ot) + np.einsum('m,mij->ij', w**2, Sqt)
            pred = w @ mq + kqt @ np.linalg.solve(Kt, yt - mt)
            out[meta] = (np.sqrt(np.mean((pred - yq) ** 2)), w)
        big = {f"w{i+1}": round(v, 2) for i, v in enumerate(out[True][1]) if v > 0.1}
        print(f"{name:20s} {k:3d} {out[True][0]:7.3f} {out[False][0]:7.3f}  {big}")
