"""Características del interior del nódulo (sección 6 del resumen de la materia).

Tabla "qué extraigo según qué distingue" de la materia:
  tamaño/forma        -> área (+ diámetro equivalente, circularidad, solidez,
                         excentricidad, fracción fina por apertura = espículas)
  borde               -> nitidez del borde (gradiente de Sobel sobre el contorno)
  brillo/densidad     -> intensidad media y desvío de la región (máscara × original),
                         más asimetría y curtosis del histograma (fuera de la materia)
  mismo brillo, distinto dibujo -> textura (GLCM + atributos de Haralick)
Más dos medidas de composición específicas de TC (fracción de calcio y de aire).

Todo en unidades físicas (mm, HU) para que sea comparable entre tomógrafos.

GLCM (Haralick, 1973), siguiendo el pipeline de la materia (6.6):
  1. reducir niveles (16, mismo rango en todos los nódulos);
  2. solo dentro de la región de interés: a los píxeles de afuera les damos un
     nivel extra (el 17º) y después tiramos su fila y columna, así solo se
     cuentan pares con los DOS píxeles dentro del nódulo;
  3. distancia 1, 4 direcciones (0°, 45°, 90°, 135°) promediadas (el tejido
     no tiene orientación preferida), simetrizada y normalizada;
  4. atributos: energía, entropía, contraste, homogeneidad, correlación,
     cluster shade y cluster prominence.

Forma y apertura: la apertura de la segmentación corta vasos pegados, pero
también "afeita" espículas finas (la materia lo advierte: la apertura borra
estructuras delgadas). Como las espículas son un signo de malignidad, la
circularidad y la solidez se miden también sobre la máscara SIN apertura
(sufijo _cruda); el informe compara las dos versiones.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import ConvexHull
from skimage import measure, morphology
from skimage.feature import graycomatrix

from . import config as C


def cuantizar(img_hu, ventana=C.VENTANA_TEXTURA_NODULO, niveles=C.NIVELES_GLCM):
    """Reparte el rango `ventana` (en HU) en `niveles` tonos 0..niveles-1."""
    lo, hi = ventana
    x = (np.clip(img_hu, lo, hi) - lo) / (hi - lo)
    return np.minimum((x * niveles).astype(np.int32), niveles - 1)


def glcm(img_hu, mascara, ventana=C.VENTANA_TEXTURA_NODULO):
    """GLCM simetrizada y normalizada, promediada en las 4 direcciones,
    contando solo pares dentro de la máscara."""
    n = C.NIVELES_GLCM
    q = np.where(mascara, cuantizar(img_hu, ventana), n).astype(np.uint8)
    P = graycomatrix(q, distances=list(C.DISTANCIAS_GLCM), angles=list(C.ANGULOS_GLCM),
                     levels=n + 1, symmetric=True, normed=False)
    P = P[:n, :n].astype(float).sum(axis=(2, 3))
    return P / P.sum() if P.sum() > 0 else P


ATRIBUTOS = ("energia", "entropia", "contraste", "homogeneidad", "correlacion",
             "cluster_shade", "cluster_prominence")


def textura_glcm(img_hu, mascara, prefijo="glcm_", ventana=C.VENTANA_TEXTURA_NODULO):
    P = glcm(img_hu, mascara, ventana)
    if P.sum() == 0:
        return {prefijo + a: np.nan for a in ATRIBUTOS}
    i, j = np.indices(P.shape)
    mi, mj = (i * P).sum(), (j * P).sum()
    si = np.sqrt(((i - mi) ** 2 * P).sum())
    sj = np.sqrt(((j - mj) ** 2 * P).sum())
    nz = P[P > 0]
    return {
        # alta = ordenada/uniforme (pocos casilleros grandes)
        prefijo + "energia": float(np.sqrt((P ** 2).sum())),
        # alta = desorden (máxima si todos los pares son igual de probables)
        prefijo + "entropia": float(-(nz * np.log2(nz)).sum()),
        # alto = grandes saltos entre vecinos (rugosa)
        prefijo + "contraste": float(((i - j) ** 2 * P).sum()),
        # alta = vecinos parecidos (cerca de la diagonal)
        prefijo + "homogeneidad": float((P / (1 + (i - j) ** 2)).sum()),
        # relación lineal entre vecino y referencia
        prefijo + "correlacion": float(((i - mi) * (j - mj) * P).sum() / (si * sj)) if si * sj > 0 else 1.0,
        # asimetría de la GLCM: + sesgo a tonos altos, - a tonos bajos
        prefijo + "cluster_shade": float(((i + j - mi - mj) ** 3 * P).sum()),
        # "picudez": estructuras marcadas
        prefijo + "cluster_prominence": float(((i + j - mi - mj) ** 4 * P).sum()),
    }


def contorno_subpixel(mascara):
    """Contorno del nódulo con precisión sub-píxel: se suaviza apenas la máscara
    (gaussiano σ=0,8 px) y se toma la isolínea 0,5 (marching squares). El área
    y el perímetro salen del MISMO polígono, así que la circularidad nunca
    supera 1 (desigualdad isoperimétrica).

    Por qué: contar píxeles de borde (skimage.measure.perimeter) subestima el
    perímetro de objetos chicos. Con un círculo perfecto de 5 mm (~3,5 px de
    radio) daba circularidad 1,25; con uno de 20 mm, 0,91. Es decir, la medida
    dependía del TAMAÑO por un error de medición, y en datos reales los
    nódulos benignos (chicos) salían con circularidad > 1. Con el contorno
    sub-píxel el mismo círculo da 0,96-0,99 en todos los tamaños."""
    m = np.pad(mascara.astype(float), 3)
    contornos = measure.find_contours(ndi.gaussian_filter(m, 0.8), 0.5)
    if not contornos:
        return None
    return max(contornos, key=len) - 3


def _area_poligono(c):
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def forma_y_tamano(mascara, pixel_mm=C.PIXEL_MM, sufijo=""):
    props = measure.regionprops(mascara.astype(np.uint8))[0]
    area = props.area * pixel_mm ** 2
    c = contorno_subpixel(mascara)
    if c is not None and len(c) >= 4:
        a_pol = _area_poligono(c)
        perim = float(np.sum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))))
        try:
            casco = ConvexHull(c).volume  # en 2D, "volume" es el área
        except Exception:
            casco = np.nan
        circularidad = 4 * np.pi * a_pol / perim ** 2 if perim > 0 else np.nan
        solidez = a_pol / casco if casco and casco > 0 else np.nan
    else:
        circularidad = solidez = np.nan
    d = {
        # 1 = círculo; lobulado o espiculado -> menor
        "circularidad" + sufijo: circularidad,
        # área / área de la envolvente convexa: entrantes del borde la bajan
        "solidez" + sufijo: solidez,
    }
    if not sufijo:
        d = {"area_mm2": area, "diametro_eq_mm": props.equivalent_diameter_area * pixel_mm, **d}
        # Excentricidad (de la elipse con los mismos momentos que la región):
        # 0 = redondo, cerca de 1 = alargado. Mide algo que la circularidad no:
        # un óvalo liso tiene circularidad alta pero excentricidad alta; un
        # círculo con espículas, al revés.
        d["excentricidad"] = props.eccentricity
    return d


def fraccion_fina(mascara_cruda, radio_px=2):
    """Construido sobre la apertura de la materia ("elimina estructuras
    delgadas preservando la forma general"): en vez de usarla solo para
    limpiar, medimos LO QUE BORRA. La fracción del área que desaparece al
    abrir con un disco de 2 px (~3 mm de diámetro) son las partes de menos
    de ~3 mm de ancho: espículas (y vasos pegados). Es una medida de
    espiculación hecha solo con morfología."""
    if mascara_cruda.sum() == 0:
        return np.nan
    abierta = ndi.binary_opening(mascara_cruda, structure=morphology.disk(radio_px))
    return float((mascara_cruda & ~abierta).sum() / mascara_cruda.sum())


def nitidez_borde(img_hu, mascara, pixel_mm=C.PIXEL_MM):
    """Construido sobre la detección de bordes (discontinuidad, materia 4.1):
    gradiente medio de HU (Sobel) en una banda sobre el contorno, en HU/mm.
    Un borde nítido (salto brusco pulmón-nódulo) da valores altos; uno
    difuso o irregular, bajos. Es la versión objetiva de la nota de
    'margen' que dan los radiólogos en LIDC."""
    borde = ndi.binary_dilation(mascara & ~ndi.binary_erosion(mascara))
    gx, gy = ndi.sobel(img_hu, 0), ndi.sobel(img_hu, 1)
    g = np.hypot(gx, gy) / (8 * pixel_mm)  # el núcleo de Sobel suma 8x el gradiente
    return float(g[borde].mean())


def caracteristicas_interior(img_hu, mascara, pixel_mm=C.PIXEL_MM, mascara_cruda=None):
    if mascara.sum() < 5:
        return {}
    v = img_hu[mascara]  # equivale a "máscara × imagen original"
    d = forma_y_tamano(mascara, pixel_mm)
    if mascara_cruda is not None and mascara_cruda.any():
        cr = forma_y_tamano(mascara_cruda, pixel_mm, sufijo="_cruda")
        d.update({k: v_ for k, v_ in cr.items() if k != "excentricidad"})
        d["frac_fina"] = fraccion_fina(mascara_cruda)
    d["nitidez_borde_hu_mm"] = nitidez_borde(img_hu, mascara, pixel_mm)
    d["hu_media"] = float(v.mean())
    d["hu_desvio"] = float(v.std(ddof=1))
    # Asimetría y curtosis del histograma de HU (fuera de la materia).
    # Por qué pensamos que podían servir: un nódulo benigno sólido suele ser
    # homogéneo -> histograma angosto y simétrico. En uno sospechoso pueden
    # aparecer zonas de necrosis o de menor densidad (cola hacia HU bajos ->
    # asimetría negativa) o focos densos (cola hacia HU altos -> positiva); y
    # pocos píxeles muy distintos del resto (calcio puntual, burbujas de aire)
    # dan colas pesadas -> curtosis alta. Es decir, describen la FORMA del
    # histograma, algo que la media y el desvío no capturan.
    # Riesgo conocido: son momentos de orden 3 y 4, muy sensibles a pocos
    # valores extremos; con nódulos de 6 mm (~60 píxeles) pueden ser ruidosos.
    # Si sirven o no lo decide la estadística (scripts/05_estadistica.py ->
    # resultados/05_comparacion.csv): AUC, p ajustado por Holm y si sigue
    # separando dentro de cada rango de tamaño.
    d["hu_asimetria"] = float(stats.skew(v))
    d["hu_curtosis"] = float(stats.kurtosis(v))  # 0 = como una normal
    d["frac_calcio"] = float((v > C.HU_CALCIO).mean())
    d["frac_aire"] = float((v < C.HU_AIRE).mean())
    d.update(textura_glcm(img_hu, mascara))
    return d


def fraccion_calcio_robusta(img_hu, mascara, umbral=C.HU_CALCIO, area_min_px=3):
    """Calcio que no se deja engañar por el ruido (propuesta, script 08).

    Problema: con ruido de baja dosis, píxeles sueltos de tejido pasan los
    200 HU por azar y un nódulo sin calcio "parece" calcificado.
    Idea, con herramientas de la materia: el calcio de verdad forma MANCHAS
    de varios píxeles; el ruido da píxeles SUELTOS.
      1. mediana 3x3 (sección 3: el filtro para valores aislados, conserva bordes);
      2. umbral > 200 HU dentro del nódulo;
      3. etiquetado y eliminar objetos chicos por área (sección 5.5): solo
         cuentan las manchas de al menos area_min_px píxeles.
    Devuelve la fracción del área del nódulo que es calcio."""
    from scipy import ndimage as ndi
    from skimage import measure
    if mascara.sum() == 0:
        return float("nan")
    cal = (ndi.median_filter(img_hu, size=3) > umbral) & mascara
    lab = measure.label(cal, connectivity=2)
    tam = np.bincount(lab.ravel())
    tam[0] = 0
    return float((tam[lab] >= area_min_px).sum() / mascara.sum())
