"""Parámetros del pipeline (valores iniciales del anteproyecto v2).

Todo lo que se ajusta con el 80 % de entrenamiento vive acá, así queda
documentado en un solo lugar qué se cambió y por qué.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos"
DIR_DICOM = DIR_DATOS / "LIDC-IDRI"      # acá se descargan los DICOM
DIR_CORTES = DIR_DATOS / "cortes"        # un .npz por nódulo (corte de trabajo)
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = RAIZ / "figuras"

SEMILLA = 42

# --- Selección de nódulos -------------------------------------------------
MAX_ESPESOR_MM = 2.5     # cortes finos: el corte de mayor área es más representativo
MIN_ANOTACIONES = 3      # ≥3 radiólogos -> consenso y concordancia con sentido
MIN_TEXTURA = 4          # mediana de textura ≥4 = sólido
MIN_DIAMETRO_MM = 6.0    # por debajo de 6 mm Lung-RADS no cambia conducta
MAX_MALIG_BENIGNO = 2    # mediana ≤2 -> benigno
MIN_MALIG_SOSPECHOSO = 4 # mediana ≥4 -> sospechoso (3 se excluye)
FRAC_ENTRENAMIENTO = 0.8

# --- Corte de trabajo -----------------------------------------------------
PIXEL_MM = 0.7           # tamaño de píxel común tras remuestrear
ROI_MM = 64.0            # recorte centrado en la semilla
CLEVEL_CONSENSO = 0.5    # píxel en consenso si lo marcó ≥50 % de los radiólogos

# --- Segmentación ---------------------------------------------------------
UMBRAL_HU = -400         # ajustar en entrenamiento entre -600 y -300
RADIO_APERTURA_PX = 1    # 1-2 px: corta vasos finos adheridos
RADIO_MAX_MM = 30.0      # freno a fugas hacia pared torácica / mediastino
RADIO_CIERRE_PULMON_MM = 30.0  # cierre del pulmón: incorpora nódulos pegados a la pared (se ajusta en 04)

# --- Características ------------------------------------------------------
HU_CALCIO = 200
# Un nódulo se considera "calcificado" si más del 2 % de su área supera
# HU_CALCIO. Se usa para el análisis "--sin-calcio" (05 y 06): el calcio es el
# signo de benignidad que los radiólogos y Lung-RADS ya usan, y la pregunta
# clínica abierta son los nódulos SIN calcio (indeterminados).
FRAC_CALCIO_MAX = 0.02
HU_AIRE = -600
# GLCM: la materia recomienda 8-16 niveles (con más, la matriz queda rala e
# inestable, y un nódulo de 6 mm tiene apenas ~60 píxeles). Usamos 16.
NIVELES_GLCM = 16
# El rango que se reparte en esos 16 niveles tiene que cubrir los valores que
# HAY en la región: con la ventana de pulmón entera [-1000, 400] cada nivel
# valdría 87 HU y todo el interior de un nódulo sólido caería en 2-3 niveles
# (textura "plana" para todos). Por eso un rango fijo por tipo de región,
# igual para todos los nódulos (condición de la materia: mismos parámetros).
VENTANA_TEXTURA_NODULO = (-500, 300)   # 50 HU por nivel; >300 (calcio) satura
VENTANA_TEXTURA_ENTORNO = (-1000, 0)   # 62,5 HU por nivel; parénquima y vasos
DISTANCIAS_GLCM = (1,)                 # textura fina -> distancia 1
ANGULOS_GLCM = (0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490614922)

# --- Entorno --------------------------------------------------------------
HU_PULMON = -400
ANILLO_INTERNO_MM = 3.0
ANILLO_EXTERNO_MM = 6.0
HU_DENSO_ENTORNO = -500

# --- Reconstrucción -------------------------------------------------------
MU_AGUA = 0.02           # mm^-1 (aprox. a ~70 keV)
N_ANGULOS_COMPLETO = 180
N_ANGULOS_REDUCIDO = 60
I0_REFERENCIA = 1e5
I0_BAJA_DOSIS = (2.5e4, 1e4)

# --- Compresión -----------------------------------------------------------
VENTANA_8BIT = (-1000, 400)
CALIDADES_JPEG = (95, 75, 50, 25)
