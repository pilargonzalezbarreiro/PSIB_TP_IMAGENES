# TPI PSIB 2026 Q2 – Biomarcadores de imagen de nódulos pulmonares sólidos

Detección temprana de cáncer de pulmón: medidas del **interior** y del **entorno** de nódulos sólidos en tomografía (LIDC-IDRI), con técnicas clásicas de procesamiento de imágenes y validación de la segmentación contra la variabilidad entre radiólogos.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No hace falta instalar un numpy viejo para pylidc: `tpi/compat.py` recrea los alias (`np.int`, `np.bool`, `np.float`) antes de importarlo.

## Cómo correrlo (en orden)

| Paso | Script | Qué hace | Necesita imágenes |
|---|---|---|---|
| 0 | `python tests/test_fantoma.py` | Prueba todo el pipeline con un fantoma sintético | No |
| 1 | `python scripts/01_seleccionar_nodulos.py --max-pacientes 80` | Contextualización + selección de nódulos con las anotaciones de pylidc (~4 min la primera vez) | No |
| 2 | `python scripts/02_descargar_dicom.py --limite 15` | Descarga solo las series seleccionadas (todas: ~11 GB; 15 series: ~2 GB) | — |
| 3 | `python scripts/03_extraer_cortes.py` | Corte de mayor área, consenso, semilla, remuestreo 0,7 mm, ROI 64 mm | Sí |
| 4 | `python scripts/04_preprocesamiento.py --casos 3` | Reconstrucción simulada (Radon, FBP, SART, baja dosis), filtros, compresión JPEG/PNG | Sí |
| 5 | `python scripts/05_caracteristicas.py --ajustar` | Segmentación, validación (umbral de aceptabilidad), medidas de interior y entorno | Sí |
| 6 | `python scripts/06_estadistica.py` | Mann-Whitney + Holm, AUC, control por tamaño (por defecto solo entrenamiento) | Sí |

Para correr las etapas 4–6 sobre el fantoma:
```bash
python scripts/04_preprocesamiento.py --carpeta tests/salida/cortes --casos 2
python scripts/05_caracteristicas.py --carpeta tests/salida/cortes --meta tests/salida/meta.csv
```

Los resultados quedan en `resultados/` (CSV) y `figuras/` (PNG). Las imágenes DICOM y los cortes `.npz` no se suben al repo.

## Estructura

```
tpi/                    módulos (cada uno explica en su docstring el porqué de cada decisión)
  config.py             todos los parámetros en un solo lugar
  compat.py             compatibilidad pylidc/numpy y ~/.pylidcrc
  datos.py              selección de nódulos, división por paciente, corte de trabajo
  segmentacion.py       crecimiento de regiones desde semilla (+ Otsu como comparación)
  validacion.py         Dice, Jaccard, concordancia entre radiólogos, aceptabilidad
  caracteristicas.py    tamaño, forma, intensidad, composición, textura GLCM
  entorno.py            máscara de pulmón y anillos 0–3 mm / 3–6 mm
  mejoramiento.py       mediana, gaussiano, Wiener
  reconstruccion.py     HU→μ, Radon, ruido Poisson, FBP, SART
  compresion.py         ventana a 8 bits, PNG/JPEG, RMSE/PSNR/SSIM
  estadistica.py        Mann-Whitney, Holm, AUC, Spearman parcial, estratos
  visualizacion.py      figuras
scripts/                01 … 06, una etapa cada uno
tests/test_fantoma.py   fantoma sintético para verificar el código
```

## Dataset

LIDC-IDRI (TCIA, CC BY 3.0). Armato SG 3rd et al., *Medical Physics* 38:915–931, 2011. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
