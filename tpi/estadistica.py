"""Comparación entre grupos (paso "comparación" del pipeline de la materia).

La materia termina en "saco números y comparo". Con ~100 nódulos y ~30
medidas hacen falta unas pocas herramientas de estadística (fuera de la
materia, explicadas en una línea cada una):

- Boxplots por grupo: la comparación visual, primero siempre.
- Mann-Whitney U: ¿la medida tiende a ser mayor en un grupo? No supone
  distribución normal (fracción de calcio, por ejemplo, es casi siempre 0).
- AUC = U / (n1·n2): probabilidad de que un sospechoso al azar tenga un valor
  mayor que un benigno al azar. 0,5 = no distingue; 1 = separa perfecto. Se
  reporta max(AUC, 1-AUC) y en qué grupo es mayor.
- Holm: con ~30 medidas, al 5 % esperaríamos ~1-2 "significativas" por pura
  casualidad; Holm ajusta los p-valores para que eso no pase.
- Control por tamaño: los radiólogos califican más maligno lo más grande, y
  muchas medidas cambian con el tamaño. Dos controles (los del anteproyecto):
    * estratos: repetir la comparación dentro de rangos de diámetro
      (6-10, 10-20, >20 mm): si sigue separando con tamaños parecidos, aporta
      algo más que el tamaño;
    * Spearman parcial: correlación entre la medida y el grupo después de
      "descontar" el diámetro (se ordenan los valores de menor a mayor,
      se les resta la parte que se explica por el diámetro y se correlaciona
      lo que queda). Cerca de 0 = lo que la medida decía del grupo era solo
      tamaño.
"""
import numpy as np
import pandas as pd
from scipy import stats

META = {"paciente", "scan_id", "nodulo", "grupo", "conjunto", "aceptable", "dice", "jaccard",
        "concordancia_radiologos", "malignidad_mediana", "textura_mediana", "diametro_medio_mm",
        "n_anotaciones", "espesor_mm", "pixel_mm", "indice_corte", "umbral_hu", "dice_otsu",
        "n_radiologos_corte", "radio_cierre_mm", "pulmon_restringe"}


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
    """Correlación de Spearman entre x e y controlando por z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(stats.pearsonr(res(rx), res(ry))[0])


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
                    auc=max(auc, 1 - auc),
                    mayor_en="sospechoso" if auc >= 0.5 else "benigno", p=p)
        if col_tamano and col_tamano in df and c != col_tamano:
            z = df[col_tamano].to_numpy(float)
            ok2 = ok & np.isfinite(z)
            if ok2.sum() > 5 and np.std(x[ok2]) > 0:
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
