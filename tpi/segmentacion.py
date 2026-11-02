"""Segmentación del nódulo: crecimiento de regiones desde una semilla.

Por qué este método (tabla de decisión de la materia, 4.12): es UN objeto,
aproximadamente homogéneo, y sabemos dónde está (la semilla) -> region
growing. Es el ejemplo que da la propia materia ("un tumor"; "nódulo en TC").
Un umbral global solo (Otsu) no alcanza: separa "tejido" de "aire" en toda la
región, pero no sabe cuál de las manchas de tejido es el nódulo (vasos,
pared, otros nódulos también son tejido).

Criterio de semejanza: la materia lo presenta como "diferencia con la
semilla ≤ tolerancia". Acá usamos "HU > T" (pertenecer al rango de densidad
de tejido blando). Es más estable porque no depende del valor de UN píxel (la
semilla puede caer sobre un punto de calcio o de ruido) y es equivalente a
"diferencia con el tejido ≤ tolerancia" con una tolerancia fija. Con un
criterio fijo, crecer desde la semilla agregando vecinos (conectividad 8) da
exactamente "la componente conexa de (img > T) que contiene a la semilla",
que se calcula con etiquetado (sección 5.5) en vez de un bucle.

¿Cómo se elige T? Parénquima ≈ -850, nódulo sólido ≈ 0. El borde es una
transición borrosa (volumen parcial); T decide dónde cortarla. Se ajusta con
el conjunto de entrenamiento (scripts/04_caracteristicas.py --ajustar).

Defecto típico del region growing (materia: "con bordes débiles se escapa"):
el nódulo toca un vaso o la pleura y la región se fuga.

El caso más grave (con datos reales, ~30 % de los nódulos): nódulos PEGADOS A
LA PARED del tórax (yuxtapleurales). Nódulo y pared tienen la misma densidad,
así que el crecimiento sigue de largo por toda la pared. Remedio de la
materia, "de lo grande a lo chico" (4.10): primero se segmenta el PULMÓN y
recién después se crece el nódulo, solo adentro del pulmón:
  1. pulmón = aire (HU < -400) que no toca el borde de la imagen (etiquetado),
     con cierre y relleno de huecos (entorno.mascara_pulmon);
  2. un nódulo pegado a la pared NO queda adentro de esa máscara: forma una
     "muesca" en el borde del pulmón. Un CIERRE con un disco grande (~30 mm,
     la técnica del "rolling ball") rellena las muescas sin meterse en la
     pared, porque la pared queda del lado convexo. Cuanto más grande el
     disco, más recto queda el "puente" sobre la muesca (con uno chico el
     puente se hunde y corta parte del nódulo); demasiado grande podría
     rellenar las concavidades del mediastino. El radio se elige con el
     conjunto de entrenamiento junto con el umbral T (04 --ajustar).
     El cierre se calcula con la transformada de distancia (dilatar = puntos a
     distancia ≤ r; erosionar = puntos a más de r del exterior): es la misma
     operación con un disco euclídeo exacto y es instantánea para cualquier r;
  3. la región del nódulo se limita a ese pulmón cerrado.
Si la semilla quedara afuera del pulmón cerrado (nódulo muy grande o pulmón
mal segmentado), se crece sin esa restricción, como antes.

Otros dos remedios, para fugas más chicas:
  - apertura (erosión -> dilatación) con disco de 1 px: corta puentes finos
    como un vaso pegado (5.3: "objetos pegados por un puente fino");
  - radio máximo desde la semilla: frena fugas grandes (pared torácica).
Por último se rellenan los huecos internos (cavidades o zonas de aire dentro
del nódulo son parte del nódulo y queremos medirlas). Se usa relleno de
huecos y no cierre porque el cierre también suavizaría el borde y taparía los
entrantes entre lóbulos, justo lo que miden la circularidad y la solidez.
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


def cierre_euclideo(mascara, r_px):
    """Cierre (dilatación -> erosión) con un disco euclídeo de radio r_px."""
    p = int(np.ceil(r_px)) + 1
    m = np.pad(mascara, p)
    dilatada = ndi.distance_transform_edt(~m) <= r_px
    cerrada = ndi.distance_transform_edt(dilatada) > r_px
    return cerrada[p:-p, p:-p]


def region_permitida(d, radio_mm=None):
    """Pulmón cerrado recortado a la región del nódulo (se calcula una vez por
    nódulo y radio, y se guarda en d). None si la semilla queda afuera."""
    radio_mm = C.RADIO_CIERRE_PULMON_MM if radio_mm is None else radio_mm
    cache = d.setdefault("_permitida", {})
    if radio_mm not in cache:
        from .datos import _recortar
        from .entorno import mascara_pulmon
        if "_pulmon" not in d:
            d["_pulmon"] = mascara_pulmon(d["corte_completo_rem"])
        cerrado = ndi.binary_fill_holes(cierre_euclideo(d["_pulmon"], radio_mm / C.PIXEL_MM))
        roi = _recortar(cerrado, tuple(d["semilla_completo"]), d["img"].shape[0], False)
        cache[radio_mm] = roi if roi[tuple(d["semilla"])] else None
    return cache[radio_mm]


def segmentar_nodulo(d, umbral=C.UMBRAL_HU, devolver_cruda=False, img=None, radio_cierre_mm=None):
    """Segmentación completa de un nódulo (lo que usa todo el pipeline)."""
    img = d["img"] if img is None else img
    return crecimiento_regiones(img, d["semilla"], umbral=umbral,
                                permitida=region_permitida(d, radio_cierre_mm),
                                devolver_cruda=devolver_cruda)


def crecimiento_regiones(img_hu, semilla, umbral=C.UMBRAL_HU,
                         radio_apertura=C.RADIO_APERTURA_PX,
                         radio_max_mm=C.RADIO_MAX_MM, pixel_mm=C.PIXEL_MM,
                         devolver_cruda=False, permitida=None):
    """Devuelve la máscara final; con devolver_cruda=True también la máscara
    antes de la apertura (para medir forma sin "afeitar" espículas).
    permitida: máscara donde se permite crecer (el pulmón cerrado)."""
    binaria = img_hu > umbral
    if permitida is not None:
        binaria &= permitida
    rr, cc = np.ogrid[:img_hu.shape[0], :img_hu.shape[1]]
    dentro = (rr - semilla[0]) ** 2 + (cc - semilla[1]) ** 2 <= (radio_max_mm / pixel_mm) ** 2
    region = _componente_con_semilla(binaria & dentro, semilla)
    cruda = ndi.binary_fill_holes(region)
    if radio_apertura > 0:
        region = ndi.binary_opening(region, structure=morphology.disk(radio_apertura))
        region = _componente_con_semilla(region, semilla)
    final = ndi.binary_fill_holes(region)
    return (final, cruda) if devolver_cruda else final


def segmentar_otsu(img_hu, semilla):
    """Comparación: umbral automático de Otsu en la región (histograma
    bimodal aire/tejido), quedándonos con la mancha de la semilla. Otsu solo
    mira el histograma, no la posición: separa dos grupos de intensidades,
    no "detecta" el nódulo."""
    t = filters.threshold_otsu(np.clip(img_hu, -1000, 400))
    region = _componente_con_semilla(img_hu > t, semilla)
    return ndi.binary_fill_holes(region), float(t)
