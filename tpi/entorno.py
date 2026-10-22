"""Entorno del nódulo: anillos de parénquima alrededor.

Por qué anillos y no "toda la ROI": queremos comparar la MISMA porción de
tejido en todos los nódulos (3 mm y 3-6 mm del borde), independientemente
del tamaño del nódulo. Las espículas, los vasos que convergen y la
retracción pleural viven justamente en esa franja.

Se intersecta con la máscara de pulmón para no medir pared torácica o
mediastino cuando el nódulo está pegado a ellos.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

from . import config as C
from .caracteristicas import intensidad, textura_glcm


def mascara_pulmon(corte_hu, pixel_mm=C.PIXEL_MM):
    """Pulmón = aire (HU < -400) que NO toca el borde de la imagen (el aire
    de afuera del cuerpo sí lo toca), componentes grandes, cierre y relleno
    para recuperar vasos y nódulos dentro del pulmón."""
    aire = corte_hu < C.HU_PULMON
    lab = measure.label(aire, connectivity=1)
    bordes = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    aire &= ~np.isin(lab, bordes)
    min_px = int(500 / pixel_mm ** 2)  # descarta componentes < 5 cm²
    lab = measure.label(aire, connectivity=1)
    tam = np.bincount(lab.ravel())
    tam[0] = 0
    aire = tam[lab] >= min_px
    r = int(round(5 / pixel_mm))
    aire = np.pad(aire, r)  # pad para que el cierre no se "coma" los bordes
    aire = ndi.binary_closing(aire, structure=morphology.disk(r))[r:-r, r:-r]
    return ndi.binary_fill_holes(aire)


def anillos(mascara_nodulo, pulmon, pixel_mm=C.PIXEL_MM,
            r_int=C.ANILLO_INTERNO_MM, r_ext=C.ANILLO_EXTERNO_MM):
    # Dilatar con un disco de radio r mm = todos los puntos a ≤ r mm del nódulo.
    # Con distancia euclídea es equivalente y más exacto que un disco discreto.
    dist = ndi.distance_transform_edt(~mascara_nodulo) * pixel_mm
    interno = (dist > 0) & (dist <= r_int) & pulmon
    externo = (dist > r_int) & (dist <= r_ext) & pulmon
    return interno, externo


def caracteristicas_entorno(img_hu, mascara_nodulo, pulmon, pixel_mm=C.PIXEL_MM):
    interno, externo = anillos(mascara_nodulo, pulmon, pixel_mm)
    d = {}
    for nombre, m in (("int", interno), ("ext", externo)):
        p = f"anillo_{nombre}_"
        d[p + "area_mm2"] = m.sum() * pixel_mm ** 2
        if m.sum() < 10:
            continue
        v = img_hu[m]
        d.update({p + "hu_media": v.mean(), p + "hu_desvio": v.std(ddof=1),
                  p + "frac_denso": float((v > C.HU_DENSO_ENTORNO).mean())})
        d.update(textura_glcm(img_hu, m, prefijo=p + "glcm_"))
    # Fracción del perímetro del nódulo que NO está rodeada de pulmón
    # (proxy de contacto con pleura/mediastino; se reporta por separado).
    borde = mascara_nodulo & ~ndi.binary_erosion(mascara_nodulo)
    vecinos = ndi.binary_dilation(borde, morphology.disk(2)) & ~mascara_nodulo
    d["frac_contacto_no_pulmon"] = float((vecinos & ~pulmon).sum() / max(vecinos.sum(), 1))
    return d, (interno, externo)
