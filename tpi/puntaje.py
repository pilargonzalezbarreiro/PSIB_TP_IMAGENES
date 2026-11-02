"""Puntaje combinado: juntar varias medidas en un solo número por nódulo.

Es el objetivo intermedio del anteproyecto y la forma de "construir un
algoritmo" encima de las medidas: una sola medida rara vez separa bien;
varias que miran cosas distintas (tamaño, borde, textura, entorno) pueden
hacerlo mejor. La materia termina en "vector de características para
clasificación"; esto es el paso siguiente, hecho de la forma más simple y
transparente posible (sin modelos de caja negra):

  1. En los nódulos de ENTRENAMIENTO, se calcula la AUC de cada medida y en
     qué grupo es mayor (su "signo").
  2. Se eligen las k medidas con mayor AUC, salteando las que repiten a una
     ya elegida (|Spearman| ≥ 0,9): no sirve sumar dos veces lo mismo.
  3. Cada medida se estandariza (z = (x - media) / desvío, con media y desvío
     de entrenamiento) para que todas pesen igual, sin importar sus unidades.
  4. Puntaje = promedio de signo · z. Alto = "se parece a los sospechosos".

Para saber si funciona sin hacer trampa, se evalúa en nódulos que NO se
usaron para elegir las medidas:
  - validación cruzada por paciente dentro de entrenamiento (k pliegues: se
    entrena con 4/5 de los pacientes y se puntúa el 1/5 restante, rotando);
  - y una sola vez, al final, en el 20 % de evaluación (--evaluar).
"""
import numpy as np
import pandas as pd
from scipy import stats


def auc(y, s):
    """AUC = P(puntaje de un sospechoso al azar > el de un benigno al azar)."""
    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float(stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic / (n1 * n0))


def curva_roc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    orden = np.argsort(-s)
    y = y[orden]
    tpr = np.concatenate([[0], np.cumsum(y == 1) / max((y == 1).sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(y == 0) / max((y == 0).sum(), 1)])
    return fpr, tpr


def ic_bootstrap(y, s, grupos, n=1000, semilla=0):
    """Intervalo de confianza 95 % de la AUC remuestreando PACIENTES."""
    rng = np.random.default_rng(semilla)
    y, s, grupos = np.asarray(y), np.asarray(s, float), np.asarray(grupos)
    pacientes = np.unique(grupos)
    idx = {p: np.flatnonzero(grupos == p) for p in pacientes}
    vals = []
    for _ in range(n):
        elegidos = np.concatenate([idx[p] for p in rng.choice(pacientes, len(pacientes))])
        a = auc(y[elegidos], s[elegidos])
        if np.isfinite(a):
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


class Puntaje:
    def __init__(self, k=5, max_corr=0.9):
        self.k, self.max_corr = k, max_corr

    def ajustar(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        filas = []
        for c in X.columns:
            x = X[c].to_numpy(float)
            if np.isfinite(x).sum() < 6 or np.nanstd(x) == 0:
                continue
            a = auc(y, x)
            if np.isfinite(a):
                filas.append((c, max(a, 1 - a), 1.0 if a >= 0.5 else -1.0))
        filas.sort(key=lambda t: -t[1])
        elegidas = []
        for c, a, signo in filas:
            if len(elegidas) == self.k:
                break
            repetida = any(abs(stats.spearmanr(X[c], X[e], nan_policy="omit").statistic) >= self.max_corr
                           for e, _, _ in elegidas)
            if not repetida:
                elegidas.append((c, a, signo))
        self.medidas = pd.DataFrame(elegidas, columns=["medida", "auc_entrenamiento", "signo"])
        cols = self.medidas["medida"].tolist()
        self.media = X[cols].mean()
        self.desvio = X[cols].std(ddof=1).replace(0, 1)
        return self

    def puntuar(self, X: pd.DataFrame):
        cols = self.medidas["medida"].tolist()
        z = (X[cols] - self.media) / self.desvio
        z = z.fillna(0.0)  # medida faltante = valor medio (no empuja hacia ningún grupo)
        return (z * self.medidas.set_index("medida")["signo"]).mean(axis=1).to_numpy()


def validacion_cruzada(X, y, pacientes, k=5, pliegues=5, semilla=0):
    """Puntajes 'fuera de pliegue': cada nódulo puntuado por un puntaje que
    se armó sin ver a ningún nódulo de su paciente."""
    y, pacientes = np.asarray(y), np.asarray(pacientes)
    unicos = np.unique(pacientes)
    rng = np.random.default_rng(semilla)
    rng.shuffle(unicos)
    pliegues = max(2, min(pliegues, len(unicos)))
    asignacion = {p: i % pliegues for i, p in enumerate(unicos)}
    pl = np.array([asignacion[p] for p in pacientes])
    s = np.full(len(y), np.nan)
    for f in range(pliegues):
        tr, te = pl != f, pl == f
        if len(np.unique(y[tr])) < 2:
            continue
        modelo = Puntaje(k).ajustar(X[tr], y[tr])
        s[te] = modelo.puntuar(X[te])
    return s
