"""Cosas compartidas por los scripts de comparaciones/.

Estos scripts NO forman parte del pipeline principal (scripts/01-05): no
alimentan las características ni la estadística. Cada uno responde una
pregunta de "¿qué método conviene y por qué?" sobre algunos nódulos.
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.caracteristicas import textura_glcm  # noqa: E402
from tpi.datos import _recortar, _remuestrear, cargar_corte  # noqa: E402
from tpi.segmentacion import segmentar_nodulo  # noqa: E402

DIR_RES = C.DIR_RESULTADOS / "comparaciones"
DIR_FIG = C.DIR_FIGURAS / "comparaciones"
DIR_CACHE = C.DIR_DATOS / "cache_reconstrucciones"
RANGO_HU = 1400.0  # ventana de pulmón [-1000, 400]
BAJA = C.I0_BAJA_DOSIS[-1]  # la dosis más baja que simulamos (1/10 de la referencia)


def argumentos(descripcion, casos_por_defecto=3):
    ap = argparse.ArgumentParser(description=descripcion)
    ap.add_argument("--casos", type=int, default=casos_por_defecto, help="cuántos nódulos usar")
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES), help="carpeta con los .npz de los cortes")
    return ap


def listar_casos(carpeta, n):
    archivos = sorted(Path(carpeta).glob("*.npz"))[:n]
    if not archivos:
        sys.exit(f"No hay cortes en {carpeta}. Corré antes scripts/03_extraer_cortes.py "
                 "(o tests/test_fantoma.py y usá --carpeta tests/salida/cortes).")
    DIR_RES.mkdir(parents=True, exist_ok=True)
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    return archivos


def original(d):
    return np.maximum(d["corte_completo"], -1000)


def umbral_segmentacion():
    ajuste = C.DIR_RESULTADOS / "04_ajuste_umbral.csv"
    if ajuste.exists():
        return float(pd.read_csv(ajuste).sort_values("dice_medio").iloc[-1]["umbral_hu"])
    return C.UMBRAL_HU


def recorte_nodulo(img, d, lado_mm=64):
    """Recorte de 64 mm alrededor del nódulo en el corte completo (para figuras)."""
    esc = C.PIXEL_MM / float(d["pixel_mm_original"])
    r, c = (np.asarray(d["semilla_completo"]) * esc).astype(int)
    h = int(lado_mm / float(d["pixel_mm_original"]) / 2)
    return img[max(r - h, 0):r + h, max(c - h, 0):c + h]


def roi_trabajo(img_completa, d):
    """Mismo remuestreo y recorte que el corte de trabajo (datos.preparar_corte)."""
    rem = _remuestrear(np.maximum(img_completa, -1024), float(d["pixel_mm_original"]), 1)
    return _recortar(rem, tuple(d["semilla_completo"]), d["img"].shape[0], -1000.0)


def mascara_nodulo(d):
    """Máscara del nódulo segmentada sobre la imagen ORIGINAL. Se usa la misma
    en todas las versiones de la imagen: así solo cambia la imagen, no la región."""
    return segmentar_nodulo(d, umbral=umbral_segmentacion(), img=roi_trabajo(original(d), d))


def medidas_nodulo(img_completa, d, mascara):
    roi = roi_trabajo(img_completa, d)
    v = roi[mascara]
    return {"hu_media": v.mean(), "hu_desvio": v.std(ddof=1), **textura_glcm(roi, mascara)}


def ruta_cache(ruta_caso):
    return DIR_CACHE / f"{Path(ruta_caso).stem}.npz"


def cargar_cache(ruta_caso):
    p = ruta_cache(ruta_caso)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


__all__ = ["C", "np", "pd", "cargar_corte"]
