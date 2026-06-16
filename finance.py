
import numpy as np
import matplotlib.pyplot as plt
 

# Parametres 
S0    = 50.0
K     = 50.0
r     = 0.05
sigma = 0.20
T     = 1.0
M     = 100_000
 
# 1) METHODE DE MONTE-CARLO NAIVE (max discret)  -- Sections 3 a 5

def price_naive(N, M=M, seed=0):
    rng  = np.random.default_rng(seed)
    dt   = T / N
    Z    = rng.standard_normal((M, N))
    incr = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    logS = np.log(S0) + np.cumsum(incr, axis=1)
    S    = np.exp(logS)                      # valeurs aux dates t1..tN
    Mmax = np.maximum(S0, S.max(axis=1))     # on inclut S0 dans le max
    disc = np.exp(-r * T) * np.maximum(Mmax - K, 0.0)
    return disc.mean(), disc.std(ddof=1) / np.sqrt(M)
 
Ns = [10, 50, 100, 250, 500, 1000]
prices, errs = [], []
for N in Ns:
    p, e = price_naive(N)
    prices.append(p); errs.append(e)
    print(f"N={N:5d}  prix={p:.4f}  (+/- {1.96*e:.4f})")
 
# Regression lineaire  prix ~ a + b/sqrt(N) ; l'ordonnee a l'origine a
# est l'estimation du prix continu (1/sqrt(N) -> 0)
x = 1.0 / np.sqrt(Ns)
b, a = np.polyfit(x, prices, 1)              # prices = b*x + a
print(f"\n[Naif] Extrapolation prix continu (1/sqrt(N)->0) = {a:.4f}")
 
# 2) CORRECTION PAR LE PONT BROWNIEN (N = 1)  -- Section 7
def price_bridge(M=M, seed=1):
    rng = np.random.default_rng(seed)
    X0  = np.log(S0)
    Z   = rng.standard_normal(M)
    XT  = X0 + (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z   # log-prix a maturite
    U   = rng.random(M)
    U   = np.clip(U, 1e-15, 1.0)             # 1-U a meme loi que U
    MX  = 0.5 * (X0 + XT + np.sqrt((XT - X0)**2 - 2 * sigma**2 * T * np.log(U)))
    Mc  = np.exp(MX)                         # maximum continu de l'actif
    disc = np.exp(-r * T) * np.maximum(Mc - K, 0.0)
    return disc.mean(), disc.std(ddof=1) / np.sqrt(M)
 
pb, eb = price_bridge()
print(f"[Pont brownien] prix = {pb:.4f}  IC95% = [{pb-1.96*eb:.4f} ; {pb+1.96*eb:.4f}]")
 
# 3) FIGURE : convergence du prix en fonction de 1/sqrt(N)
xx = np.linspace(0, max(x) * 1.05, 100)
plt.figure(figsize=(8, 5))
plt.errorbar(x, prices, yerr=[1.96 * e for e in errs], fmt='o',
             capsize=4, label="Prix Monte-Carlo (max discret)", color="#1f4e79")
plt.plot(xx, a + b * xx, '--', color="#c0392b",
         label=f"Regression : {a:.3f} + ({b:.3f})/\u221aN")
plt.scatter([0], [a], color="#c0392b", zorder=5)
plt.annotate(f"Prix continu extrapole = {a:.4f}", (0, a),
             textcoords="offset points", xytext=(15, -15))
plt.xlabel(r"$1/\sqrt{N}$")
plt.ylabel("Prix de l'option Lookback")
plt.title("Convergence du prix en fonction du pas de discretisation")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
