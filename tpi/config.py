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

# --- Características ------------------------------------------------------
HU_CALCIO = 200
HU_AIRE = -600
VENTANA_TEXTURA = (-1000, 400)
NIVELES_GLCM = 32
DISTANCIAS_GLCM = (1, 2)
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
