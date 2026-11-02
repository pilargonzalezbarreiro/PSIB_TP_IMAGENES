"""Mejoramiento: filtros de reducción de ruido (sección 3 de la materia).

Regla de la materia: el filtro se elige según el TIPO de ruido.
- Mediana 3x3 (no lineal): para ruido impulsivo (sal y pimienta); preserva bordes.
- Gaussiano σ=1 px (lineal, pasa bajos): para ruido gaussiano / granulado.
- Wiener 5x5 (adaptativo): suaviza mucho donde la zona es pareja (varianza
  local baja = probablemente ruido) y poco donde hay bordes (varianza local
  alta = probablemente estructura). Compromiso entre sacar ruido y conservar
  bordes.

El ruido de una tomografía reconstruida (y más el de baja dosis) es
GRANULADO y está en toda la imagen, no son píxeles sueltos: a priori le
corresponde el gaussiano; Wiener promete conservar mejor los bordes. Se
comparan los tres con números en comparaciones/filtros.py.

Se evalúan sobre una imagen con ruido conocido (la reconstrucción de baja
dosis) contra la original, así "mejor" deja de ser una impresión visual.

Importante (materia): el filtrado cambia la textura. Por eso estos filtros
NO se aplican antes de medir las características: las medidas se hacen sobre
la imagen original, igual para todos los nódulos.
"""
import numpy as np
from scipy import ndimage as ndi


def mediana(img, tam=3):
    return ndi.median_filter(img, size=tam)


def gaussiano(img, sigma=1.0):
    return ndi.gaussian_filter(img, sigma=sigma)


def filtro_wiener(img, ventana=5, ruido=None):
    """Misma fórmula que scipy.signal.wiener, pero sin dividir por cero en
    zonas perfectamente uniformes (el aire fuera del cuerpo, -1000 HU):
        salida = m + max(v - σ², 0) / max(v, σ²) · (img - m)
    m, v = media y varianza locales en la ventana; σ² = varianza del ruido
    (por defecto, el promedio de las varianzas locales).
    Donde v ≈ σ² (zona pareja) la salida ≈ m (promedia); donde v >> σ²
    (borde) la salida ≈ img (no toca)."""
    x = img.astype(np.float64)
    m = ndi.uniform_filter(x, ventana)
    v = np.maximum(ndi.uniform_filter(x ** 2, ventana) - m ** 2, 0)
    s2 = v.mean() if ruido is None else ruido
    return (m + np.maximum(v - s2, 0) / np.maximum(v, s2) * (x - m)).astype(img.dtype)


FILTROS = {"mediana_3x3": mediana, "gaussiano_s1": gaussiano, "wiener_5x5": filtro_wiener}
