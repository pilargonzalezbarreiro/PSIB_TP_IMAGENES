"""Entorno del nódulo: anillos de parénquima alrededor.

Por qué anillos de ancho fijo en mm: queremos comparar la MISMA franja de
tejido (0-3 mm y 3-6 mm del borde) en nódulos chicos y grandes. Las
espículas, los vasos que convergen y la retracción pleural viven ahí.

Construcción (dilatación, sección 5.2 de la materia: "margen alrededor de un
tumor"):
  anillo interno = dilatación(nódulo, disco 3 mm) - nódulo
  anillo externo = dilatación(nódulo, disco 6 mm) - dilatación(nódulo, disco 3 mm)
Dilatar con un disco de radio r es, por definición, quedarse con todos los
puntos a distancia ≤ r del objeto. Se calcula con la transformada de
distancia (distancia euclídea exacta de cada píxel al nódulo, en mm) en vez de
con un disco de píxeles: con píxeles de 0,7 mm el disco discreto redondea
3 mm a 2,8 y 6 mm a 6,3, y los anillos cambian un 11-17 % de tamaño. Es la
misma operación, sin el error del redondeo. Se intersecta con la máscara de pulmón para no medir pared torácica
o mediastino cuando el nódulo está pegado a ellos.

La textura del entorno se calcula con la misma GLCM que el interior, pero con
un rango de cuantización propio del parénquima ([-1000, 0] HU).

Estructuras que convergen (construido sobre la esqueletización de la
materia, 5.4: "vasos: esqueleto -> largo, ramificaciones"): las estructuras
densas del entorno (vasos, espículas, tractos) se reducen a su esqueleto
(líneas de 1 píxel centradas). Con eso se mide:
  - longitud total de esas líneas por mm de borde del nódulo;
  - cuántas de ellas TOCAN el nódulo (llegan hasta su borde): el signo
    radiológico de "convergencia vascular" o de espículas que se extienden.
Un nódulo benigno suele estar "suelto" en el parénquima; uno maligno tiende
a traccionar y atraer estructuras.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.morphology import skeletonize

from . import config as C
from .caracteristicas import textura_glcm


def mascara_pulmon(corte_hu, pixel_mm=C.PIXEL_MM):
    """Pulmón, "de lo grande a lo chico" (materia 4.10):
      1. umbral binario invertido: aire = HU < -400 (los pulmones son lo oscuro);
      2. etiquetado: descartar el aire que toca el borde de la imagen (es el
         de afuera del cuerpo) y las manchas chicas (< 5 cm², por área);
      3. cierre + relleno de huecos para recuperar vasos y nódulos, que son
         densos y quedaron como agujeros dentro del pulmón."""
    aire = corte_hu < C.HU_PULMON
    lab = measure.label(aire, connectivity=1)
    bordes = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    aire &= ~np.isin(lab, bordes)
    min_px = int(500 / pixel_mm ** 2)
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
    dist = ndi.distance_transform_edt(~mascara_nodulo) * pixel_mm  # mm al nódulo
    interno = (dist > 0) & (dist <= r_int) & pulmon     # = dilatación 3 mm - nódulo
    externo = (dist > r_int) & (dist <= r_ext) & pulmon  # = dilatación 6 mm - dilatación 3 mm
    return interno, externo


def caracteristicas_entorno(img_hu, mascara_nodulo, pulmon, pixel_mm=C.PIXEL_MM):
    interno, externo = anillos(mascara_nodulo, pulmon, pixel_mm)
    d = {}
    for nombre, m in (("int", interno), ("ext", externo)):
        p = f"anillo_{nombre}_"
        if m.sum() < 10:
            continue
        v = img_hu[m]
        d.update({p + "hu_media": float(v.mean()), p + "hu_desvio": float(v.std(ddof=1)),
                  # fracción de estructuras densas (vasos, espículas, tractos)
                  p + "frac_denso": float((v > C.HU_DENSO_ENTORNO).mean())})
        d.update(textura_glcm(img_hu, m, prefijo=p + "glcm_", ventana=C.VENTANA_TEXTURA_ENTORNO))
    d.update(estructuras_convergentes(img_hu, mascara_nodulo, pulmon, pixel_mm))
    # Fracción del borde del nódulo que NO está rodeada de pulmón
    # (proxy de contacto con pleura/mediastino).
    borde = mascara_nodulo & ~ndi.binary_erosion(mascara_nodulo)
    vecinos = ndi.binary_dilation(borde, morphology.disk(2)) & ~mascara_nodulo
    d["frac_contacto_no_pulmon"] = float((vecinos & ~pulmon).sum() / max(vecinos.sum(), 1))
    return d, (interno, externo)


def estructuras_convergentes(img_hu, mascara_nodulo, pulmon, pixel_mm=C.PIXEL_MM,
                             r_mm=C.ANILLO_EXTERNO_MM):
    dist = ndi.distance_transform_edt(~mascara_nodulo) * pixel_mm
    zona = (dist > 0) & (dist <= r_mm) & pulmon
    densas = (img_hu > C.HU_DENSO_ENTORNO) & zona
    # sacar puntitos sueltos (ruido) por área, como en la materia (etiquetado)
    lab = measure.label(densas, connectivity=2)
    tam = np.bincount(lab.ravel())
    tam[0] = 0
    densas = tam[lab] >= 4
    esq = skeletonize(densas)
    perimetro = max(measure.perimeter(mascara_nodulo) * pixel_mm, 1e-6)
    # una estructura "toca" el nódulo si su esqueleto llega a menos de 1,5 px del borde
    lab_esq = measure.label(esq, connectivity=2)
    toca = np.unique(lab_esq[esq & (dist <= 1.5 * pixel_mm)])
    n_toca = int((toca > 0).sum())
    return {"esqueleto_mm_por_mm_borde": float(esq.sum() * pixel_mm / perimetro),
            "n_estructuras_convergentes": n_toca,
            "convergentes_por_10mm_borde": float(10 * n_toca / perimetro)}
