"""Segmentación semiautomática del nódulo a partir de una semilla.

Crecimiento de regiones con criterio de umbral fijo: arrancando en la
semilla, se agregan vecinos (conectividad 8) mientras tengan HU > T. Con un
criterio fijo, eso es exactamente "la componente conexa de (img > T) que
contiene a la semilla", y así lo implementamos (más rápido y sin bucles).

¿Por qué T ≈ -400 HU? El parénquima aireado está cerca de -850 HU y un
nódulo sólido cerca de 0-50 HU; -400 queda en el medio del borde (efecto de
volumen parcial), así que el contorno cae aproximadamente donde lo dibuja
un radiólogo. Se ajusta en entrenamiento.

Problema típico: el nódulo toca un vaso o la pleura -> la componente se
"fuga". La apertura morfológica corta conexiones finas (vasos) y el radio
máximo frena fugas grandes (pared torácica).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from . import config as C


def _componente_con_semilla(binaria, semilla):
    etiquetas = measure.label(binaria, connectivity=2)
    lab = etiquetas[semilla]
    if lab == 0:
        # La semilla quedó afuera (p. ej. la borró la apertura): tomar la
        # componente más cercana a la semilla.
        if etiquetas.max() == 0:
            return np.zeros_like(binaria, dtype=bool)
        dist = ndi.distance_transform_edt(etiquetas == 0, return_indices=True)[1]
        lab = etiquetas[dist[0][semilla], dist[1][semilla]]
    return etiquetas == lab


def crecimiento_regiones(img_hu, semilla, umbral=C.UMBRAL_HU,
                         radio_apertura=C.RADIO_APERTURA_PX,
                         radio_max_mm=C.RADIO_MAX_MM, pixel_mm=C.PIXEL_MM):
    binaria = img_hu > umbral
    # Restricción espacial: nada a más de radio_max_mm de la semilla.
    rr, cc = np.ogrid[:img_hu.shape[0], :img_hu.shape[1]]
    dentro = (rr - semilla[0]) ** 2 + (cc - semilla[1]) ** 2 <= (radio_max_mm / pixel_mm) ** 2
    region = _componente_con_semilla(binaria & dentro, semilla)
    if radio_apertura > 0:
        region = ndi.binary_opening(region, structure=morphology.disk(radio_apertura))
        region = _componente_con_semilla(region, semilla)
    return ndi.binary_fill_holes(region)


def segmentar_otsu(img_hu, semilla):
    """Comparación: umbral automático de Otsu dentro de la ROI."""
    t = filters.threshold_otsu(np.clip(img_hu, -1000, 400))
    region = _componente_con_semilla(img_hu > t, semilla)
    return ndi.binary_fill_holes(region), float(t)
