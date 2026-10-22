"""Comparación entre grupos sin modelos entrenados.

- Mann-Whitney U: no asume normalidad (con ~100 nódulos y medidas como la
  fracción de calcio, muy asimétricas, una t de Student no corresponde).
- AUC = U / (n1·n2): probabilidad de que un sospechoso al azar tenga un
  valor mayor que un benigno al azar. 0,5 = no discrimina. Si da < 0,5 la
  medida discrimina "al revés" (más baja en sospechosos): reportamos la
  dirección y max(AUC, 1-AUC).
- Holm: con ~40 medidas, al 5 % esperaríamos ~2 "significativas" por azar.
  Holm controla la probabilidad de al menos un falso positivo.
- Control de tamaño: los radiólogos califican más maligno lo más grande, así
  que muchas medidas pueden "funcionar" solo porque correlacionan con el
  diámetro. La Spearman parcial mide la asociación medida-grupo una vez
  descontado el diámetro.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

META = {"paciente", "scan_id", "nodulo", "grupo", "conjunto", "aceptable", "dice", "jaccard",
        "concordancia_radiologos", "malignidad_mediana", "textura_mediana", "diametro_medio_mm",
        "n_anotaciones", "espesor_mm", "pixel_mm", "indice_corte", "umbral_hu", "dice_otsu"}


def holm(p):
    p = np.asarray(p, float)
    orden = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acum = 0.0
    for r, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - r) * p[i]))
        ajust[i] = acum
    return ajust


def spearman_parcial(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return stats.pearsonr(res(rx), res(ry))[0]


def comparar_grupos(df, columnas=None, col_tamano="diametro_eq_mm"):
    df = df[df["grupo"].isin(["benigno", "sospechoso"])]
    if columnas is None:
        columnas = [c for c in df.select_dtypes("number").columns if c not in META]
    y = (df["grupo"] == "sospechoso").to_numpy(int)
    filas = []
    for c in columnas:
        x = df[c].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() < 6 or len(np.unique(y[ok])) < 2:
            continue
        s, b = x[ok & (y == 1)], x[ok & (y == 0)]
        u, p = stats.mannwhitneyu(s, b, alternative="two-sided")
        auc = u / (len(s) * len(b))
        fila = dict(medida=c, n_sosp=len(s), n_benig=len(b),
                    mediana_sosp=np.median(s), mediana_benig=np.median(b),
                    auc=max(auc, 1 - auc), mayor_en="sospechoso" if auc >= 0.5 else "benigno",
                    p=p)
        if col_tamano in df and c != col_tamano:
            z = df[col_tamano].to_numpy(float)
            ok2 = ok & np.isfinite(z)
            fila["spearman_parcial_vs_tamano"] = spearman_parcial(x[ok2], y[ok2], z[ok2])
        filas.append(fila)
    res = pd.DataFrame(filas)
    if len(res):
        res["p_holm"] = holm(res["p"])
        res = res.sort_values("auc", ascending=False).reset_index(drop=True)
    return res


def por_estratos(df, columnas, col_tamano="diametro_eq_mm", cortes=(6, 10, 20, np.inf)):
    df = df.copy()
    df["estrato"] = pd.cut(df[col_tamano], bins=list(cortes), right=False)
    salidas = []
    for e, sub in df.groupby("estrato", observed=True):
        r = comparar_grupos(sub, columnas, col_tamano=None)
        if len(r):
            r.insert(0, "estrato_mm", str(e))
            salidas.append(r)
    return pd.concat(salidas, ignore_index=True) if salidas else pd.DataFrame()
