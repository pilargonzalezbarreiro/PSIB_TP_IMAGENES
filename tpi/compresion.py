"""Compresión y descompresión, y métricas de calidad de imagen.

Las tomografías tienen 12 bits (HU de -1024 a ~3000); JPEG baseline solo
acepta 8 bits. Por eso primero aplicamos una ventana [-1000, 400] HU y la
llevamos a 0-255: cada nivel ≈ 5,5 HU. Esa cuantización ya es una pérdida
(se reporta aparte: PNG sobre la imagen de 8 bits la mide).

JPEG divide en bloques 8x8, aplica DCT y cuantiza más fuerte las altas
frecuencias -> a calidad baja aparecen bloques y se borran texturas finas,
justo lo que miden las características de textura.
"""
import io

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import config as C


def hu_a_8bits(hu, ventana=C.VENTANA_8BIT):
    lo, hi = ventana
    return np.round((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def de_8bits_a_hu(img8, ventana=C.VENTANA_8BIT):
    lo, hi = ventana
    return img8.astype(np.float32) / 255 * (hi - lo) + lo


def comprimir(img8, formato="JPEG", calidad=75):
    """Devuelve (imagen descomprimida uint8, bytes del archivo)."""
    buf = io.BytesIO()
    kw = {"quality": int(calidad)} if formato == "JPEG" else {"optimize": True}
    Image.fromarray(img8).save(buf, format=formato, **kw)
    n = buf.tell()
    buf.seek(0)
    return np.array(Image.open(buf)), n


def metricas(referencia, prueba, rango, mascara=None):
    ref = np.asarray(referencia, float)
    pru = np.asarray(prueba, float)
    if mascara is not None:
        ref, pru = np.where(mascara, ref, 0), np.where(mascara, pru, 0)
    err = (ref - pru)[mascara] if mascara is not None else (ref - pru)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return dict(
        rmse=rmse,
        psnr=float(peak_signal_noise_ratio(ref, pru, data_range=rango)) if rmse > 0 else np.inf,
        ssim=float(structural_similarity(ref, pru, data_range=rango)),
    )
