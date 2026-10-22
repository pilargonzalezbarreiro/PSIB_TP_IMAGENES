"""Medidas del interior del nódulo (tamaño, forma, intensidad, composición,
textura). Todas en unidades físicas (mm, HU) para que sean comparables
entre tomógrafos con distinto tamaño de píxel.

Textura (GLCM, matriz de coocurrencia de Haralick): cuenta cuántas veces un
nivel de gris i aparece a distancia d y ángulo θ de un nivel j. De esa
matriz salen números que resumen "cómo cambia la intensidad de un píxel al
vecino": contraste alto = cambios bruscos; homogeneidad alta = vecinos
parecidos; energía alta = pocos patrones repetidos; entropía alta = desorden.

Truco de la máscara: graycomatrix trabaja sobre un rectángulo. Para contar
solo pares DENTRO de la región, a los píxeles de afuera les damos un nivel
extra (el 33) y después tiramos su fila y columna.
"""
import numpy as np
from scipy import stats
from skimage import measure
from skimage.feature import graycomatrix

from . import config as C


def cuantizar(img_hu, ventana=C.VENTANA_TEXTURA, niveles=C.NIVELES_GLCM):
    lo, hi = ventana
    x = (np.clip(img_hu, lo, hi) - lo) / (hi - lo)
    return np.minimum((x * niveles).astype(np.int32), niveles - 1)


def textura_glcm(img_hu, mascara, prefijo="glcm_"):
    q = cuantizar(img_hu)
    q = np.where(mascara, q, C.NIVELES_GLCM).astype(np.uint8)
    P = graycomatrix(q, distances=list(C.DISTANCIAS_GLCM), angles=list(C.ANGULOS_GLCM),
                     levels=C.NIVELES_GLCM + 1, symmetric=True, normed=False)
    P = P[:C.NIVELES_GLCM, :C.NIVELES_GLCM].astype(float).sum(axis=(2, 3))  # promedio d, θ
    if P.sum() == 0:
        return {prefijo + k: np.nan for k in
                ("contraste", "disimilitud", "homogeneidad", "energia", "correlacion", "entropia")}
    P /= P.sum()
    i, j = np.indices(P.shape)
    mi, mj = (i * P).sum(), (j * P).sum()
    si = np.sqrt(((i - mi) ** 2 * P).sum())
    sj = np.sqrt(((j - mj) ** 2 * P).sum())
    nz = P[P > 0]
    return {
        prefijo + "contraste": float(((i - j) ** 2 * P).sum()),
        prefijo + "disimilitud": float((np.abs(i - j) * P).sum()),
        prefijo + "homogeneidad": float((P / (1 + (i - j) ** 2)).sum()),
        prefijo + "energia": float(np.sqrt((P ** 2).sum())),
        prefijo + "correlacion": float(((i - mi) * (j - mj) * P).sum() / (si * sj)) if si * sj > 0 else 1.0,
        prefijo + "entropia": float(-(nz * np.log2(nz)).sum()),
    }


def intensidad(valores, prefijo=""):
    v = np.asarray(valores, float)
    if v.size < 3:
        return {}
    h = np.bincount(cuantizar(v), minlength=C.NIVELES_GLCM) / v.size
    h = h[h > 0]
    return {
        prefijo + "hu_media": v.mean(), prefijo + "hu_desvio": v.std(ddof=1),
        prefijo + "hu_asimetria": float(stats.skew(v)),
        prefijo + "hu_curtosis": float(stats.kurtosis(v)),
        prefijo + "hu_p10": float(np.percentile(v, 10)),
        prefijo + "hu_p90": float(np.percentile(v, 90)),
        prefijo + "hu_entropia": float(-(h * np.log2(h)).sum()),
    }


def forma_y_tamano(mascara, pixel_mm=C.PIXEL_MM):
    props = measure.regionprops(mascara.astype(np.uint8))[0]
    area = props.area * pixel_mm ** 2
    perim = measure.perimeter(mascara) * pixel_mm
    return {
        "area_mm2": area,
        "diametro_eq_mm": props.equivalent_diameter_area * pixel_mm,
        "feret_max_mm": props.feret_diameter_max * pixel_mm,
        "perimetro_mm": perim,
        # 1 = círculo perfecto; bordes espiculados/lobulados -> menor
        "circularidad": 4 * np.pi * area / perim ** 2 if perim > 0 else np.nan,
        # área / envolvente convexa: entrantes del borde la bajan
        "solidez": props.solidity,
        "excentricidad": props.eccentricity,
    }


def caracteristicas_interior(img_hu, mascara, pixel_mm=C.PIXEL_MM):
    if mascara.sum() < 5:
        return {}
    v = img_hu[mascara]
    d = forma_y_tamano(mascara, pixel_mm)
    d.update(intensidad(v))
    d["frac_calcio"] = float((v > C.HU_CALCIO).mean())
    d["frac_aire"] = float((v < C.HU_AIRE).mean())
    d.update(textura_glcm(img_hu, mascara))
    return d
