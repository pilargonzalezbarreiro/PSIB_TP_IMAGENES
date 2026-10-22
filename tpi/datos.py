"""Selección de nódulos (pylidc) y extracción del corte de trabajo.

Dos cosas importantes que conviene entender:

1. pylidc trae TODAS las anotaciones (contornos y calificaciones de los
   radiólogos) en una base SQLite incluida en el paquete. Por eso la
   selección de nódulos corre sin haber descargado ninguna imagen: recién
   después bajamos solo las series que sobreviven a los filtros.

2. Cada radiólogo marcó los nódulos por su cuenta, así que para un mismo
   nódulo hay hasta 4 anotaciones. scan.cluster_annotations() las agrupa por
   cercanía espacial: cada grupo = un nódulo físico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import config as C
from .compat import importar_pylidc


# ---------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------
def _grupo(malig_mediana: float) -> str | None:
    if malig_mediana <= C.MAX_MALIG_BENIGNO:
        return "benigno"
    if malig_mediana >= C.MIN_MALIG_SOSPECHOSO:
        return "sospechoso"
    return None  # mediana 3 (o 2.5 / 3.5): zona gris, se excluye


def tabla_candidatos() -> pd.DataFrame:
    """Recorre LIDC con los criterios del anteproyecto y devuelve una fila por
    nódulo, incluidos los excluidos (columna `motivo_exclusion`), para poder
    reportar el diagrama de flujo de selección."""
    pl = importar_pylidc()
    filas = []
    for scan in pl.query(pl.Scan).all():
        espesor_ok = scan.slice_thickness <= C.MAX_ESPESOR_MM
        for i, anns in enumerate(scan.cluster_annotations()):
            malig = float(np.median([a.malignancy for a in anns]))
            textura = float(np.median([a.texture for a in anns]))
            diam = float(np.mean([a.diameter for a in anns]))
            grupo = _grupo(malig)
            if not espesor_ok:
                motivo = "espesor > 2.5 mm"
            elif len(anns) > 4:
                motivo = "agrupación ambigua (> 4 anotaciones)"
            elif len(anns) < C.MIN_ANOTACIONES:
                motivo = "< 3 radiólogos"
            elif textura < C.MIN_TEXTURA:
                motivo = "no sólido"
            elif diam < C.MIN_DIAMETRO_MM:
                motivo = "< 6 mm"
            elif grupo is None:
                motivo = "malignidad intermedia"
            else:
                motivo = ""
            filas.append(dict(
                paciente=scan.patient_id, scan_id=scan.id, nodulo=i,
                study_uid=scan.study_instance_uid,
                series_uid=scan.series_instance_uid,
                espesor_mm=scan.slice_thickness, pixel_mm=scan.pixel_spacing,
                n_anotaciones=len(anns), malignidad_mediana=malig,
                textura_mediana=textura, diametro_medio_mm=diam,
                grupo=grupo, motivo_exclusion=motivo,
                incluido=motivo == "",
            ))
    return pd.DataFrame(filas)


def dividir_por_paciente(df: pd.DataFrame, frac=C.FRAC_ENTRENAMIENTO,
                         semilla=C.SEMILLA) -> pd.DataFrame:
    """Divide por PACIENTE (no por nódulo): si dos nódulos del mismo paciente
    caen uno en cada conjunto, el test deja de ser independiente.
    Estratificada: los pacientes se separan según su grupo mayoritario, así
    los dos conjuntos tienen proporciones parecidas de sospechosos."""
    rng = np.random.default_rng(semilla)
    mayoria = df.groupby("paciente")["grupo"].agg(lambda g: (g == "sospechoso").mean() >= 0.5)
    train = set()
    for _, pacs in mayoria.groupby(mayoria):
        p = np.array(sorted(pacs.index))
        rng.shuffle(p)
        train |= set(p[:int(round(frac * len(p)))])
    df = df.copy()
    df["conjunto"] = np.where(df["paciente"].isin(train), "entrenamiento", "evaluacion")
    return df


# ---------------------------------------------------------------------------
# Corte de trabajo
# ---------------------------------------------------------------------------
def _remuestrear(img, pixel_mm, orden):
    factor = pixel_mm / C.PIXEL_MM
    return ndi.zoom(img.astype(np.float32 if orden else np.uint8), factor, order=orden)


def _recortar(img, centro, lado_px, relleno):
    """Recorte cuadrado centrado; rellena si se sale de la imagen."""
    r0 = int(centro[0]) - lado_px // 2
    c0 = int(centro[1]) - lado_px // 2
    salida = np.full((lado_px, lado_px), relleno, dtype=img.dtype)
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + lado_px, img.shape[0]), min(c0 + lado_px, img.shape[1])
    salida[rs - r0:re - r0, cs - c0:ce - c0] = img[rs:re, cs:ce]
    return salida


def semilla_desde_mascara(mascara) -> tuple[int, int]:
    """Píxel DE la máscara más cercano a su centroide (en nódulos con forma de
    medialuna el centroide puede caer afuera)."""
    rr, cc = np.nonzero(mascara)
    cr, cc_ = rr.mean(), cc.mean()
    k = np.argmin((rr - cr) ** 2 + (cc - cc_) ** 2)
    return int(rr[k]), int(cc[k])


def preparar_corte(corte_hu, pixel_mm, mascara_consenso, mascaras_radiologos):
    """A partir del corte completo (resolución original) y las máscaras en
    coordenadas del corte completo, devuelve todo listo para segmentar:
    remuestreo a PIXEL_MM y recorte ROI_MM x ROI_MM centrado en la semilla.

    Se separa de la lectura de pylidc para poder probarlo con un fantoma."""
    corte_hu = np.maximum(corte_hu, -1024)  # fuera del FOV algunos equipos ponen -2048
    img = _remuestrear(corte_hu, pixel_mm, 1)
    cons = _remuestrear(mascara_consenso, pixel_mm, 0).astype(bool)
    rads = [_remuestrear(m, pixel_mm, 0).astype(bool) for m in mascaras_radiologos]

    semilla_full = semilla_desde_mascara(cons)
    lado = int(round(C.ROI_MM / C.PIXEL_MM))
    roi = lambda x, rel: _recortar(x, semilla_full, lado, rel)  # noqa: E731
    img_roi = roi(img, -1000.0)
    cons_roi = roi(cons, False)
    return dict(
        img=img_roi.astype(np.float32),
        consenso=cons_roi,
        radiologos=np.stack([roi(m, False) for m in rads]),
        semilla=semilla_desde_mascara(cons_roi),
        corte_completo=corte_hu.astype(np.float32),   # para reconstrucción/compresión
        corte_completo_rem=img.astype(np.float32),    # para máscara de pulmón
        semilla_completo=np.array(semilla_full),
        pixel_mm_original=float(pixel_mm),
    )


def extraer_corte(scan, anns) -> dict:
    """Lee el volumen de un scan y arma el corte de trabajo de un nódulo."""
    from pylidc.utils import consensus

    vol = scan.to_volume(verbose=False)
    cmask, cbbox, masks = consensus(anns, clevel=C.CLEVEL_CONSENSO, pad=None)
    areas = cmask.sum(axis=(0, 1))
    k_local = int(np.argmax(areas))
    k = cbbox[2].start + k_local

    def a_corte_completo(m3d):
        m = np.zeros(vol.shape[:2], dtype=bool)
        m[cbbox[0], cbbox[1]] = m3d[:, :, k_local]
        return m

    datos = preparar_corte(
        vol[:, :, k], float(scan.pixel_spacing),
        a_corte_completo(cmask), [a_corte_completo(m) for m in masks],
    )
    datos["indice_corte"] = k
    return datos


def guardar_corte(ruta, datos: dict):
    np.savez_compressed(ruta, **datos)


def cargar_corte(ruta) -> dict:
    with np.load(ruta) as z:
        d = {k: z[k] for k in z.files}
    d["semilla"] = tuple(int(v) for v in d["semilla"])
    return d
