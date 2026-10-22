"""Mejoramiento: filtros clásicos de reducción de ruido.

- Mediana 3x3: no lineal, quita ruido impulsivo y respeta bordes.
- Gaussiano σ=1 px: lineal, pasa-bajos; suaviza ruido pero también bordes.
- Wiener 5x5: adaptativo; suaviza más donde la varianza local es baja
  (parénquima homogéneo) y menos en bordes (varianza alta).

Para evaluarlos con sentido, se aplican sobre una imagen con ruido conocido
(la reconstrucción de baja dosis) y se compara contra la de referencia.
"""
import numpy as np
from scipy import ndimage as ndi


def mediana(img, tam=3):
    return ndi.median_filter(img, size=tam)


def gaussiano(img, sigma=1.0):
    return ndi.gaussian_filter(img, sigma=sigma)


def filtro_wiener(img, ventana=5, ruido=None):
    """Wiener adaptativo (misma fórmula que scipy.signal.wiener, pero sin
    dividir por cero en zonas perfectamente uniformes, como el aire = -1000):
        salida = m + max(v - σ², 0) / max(v, σ²) · (img - m)
    m, v: media y varianza locales; σ²: varianza del ruido (por defecto, la
    media de las varianzas locales)."""
    x = img.astype(np.float64)
    m = ndi.uniform_filter(x, ventana)
    v = np.maximum(ndi.uniform_filter(x ** 2, ventana) - m ** 2, 0)
    s2 = v.mean() if ruido is None else ruido
    return (m + np.maximum(v - s2, 0) / np.maximum(v, s2) * (x - m)).astype(img.dtype)


FILTROS = {"mediana_3x3": mediana, "gaussiano_s1": gaussiano, "wiener_5x5": filtro_wiener}
