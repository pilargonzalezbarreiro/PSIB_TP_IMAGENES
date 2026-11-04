# TPI PSIB 2026 Q2 – Biomarcadores de imagen de nódulos pulmonares sólidos

Detección temprana de cáncer de pulmón: medidas del **interior** y del **entorno** de nódulos sólidos en tomografía (LIDC-IDRI), con técnicas clásicas de procesamiento de imágenes y validación de la segmentación contra la variabilidad entre radiólogos.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No hace falta instalar un numpy viejo para pylidc: `tpi/compat.py` recrea los alias (`np.int`, `np.bool`, `np.float`) antes de importarlo.

## Dos partes

- **`scripts/` — pipeline principal.** Lo que produce los resultados del TP, en orden. No filtra ni comprime nada: mide sobre la imagen original.
- **`comparaciones/` — preguntas de "¿qué método conviene?".** Cada archivo responde una (reconstrucción, filtros, compresión, robustez). No alimentan al pipeline; se corren cuando querés probar algo.

## Pipeline principal (en orden)

| Paso | Script | Qué hace | Necesita imágenes |
|---|---|---|---|
| 0 | `python tests/test_fantoma.py` | Prueba todo con un fantoma sintético | No |
| 1 | `python scripts/01_seleccionar_nodulos.py --max-pacientes 80` | Contextualización + selección de nódulos con las anotaciones de pylidc (~4 min la primera vez) | No |
| 2 | `python scripts/02_descargar_dicom.py --limite 15` | Descarga solo las series seleccionadas (todas: ~11 GB; 15 series: ~2 GB) | — |
| 3 | `python scripts/03_extraer_cortes.py` | Corte de mayor área, consenso, semilla, remuestreo 0,7 mm, ROI 64 mm | Sí |
| 4 | `python scripts/04_caracteristicas.py --ajustar` | Análisis de histogramas, segmentación, validación (umbral de aceptabilidad), medidas de interior y entorno | Sí |
| 1b | `python scripts/01_seleccionar_nodulos.py --ampliar-tamano 8 16` | Agrega a la muestra los nódulos elegibles de 8 a 16 mm (control por tamaño) | No |
| 2–3 | `python scripts/02_descargar_dicom.py` y `python scripts/03_extraer_cortes.py --borrar-dicom` | Bajan solo lo que falta extraer; `--borrar-dicom` borra cada tomografía apenas se extraen sus nódulos | — |
| 5 | `python scripts/05_estadistica.py` | Medida por medida: boxplots, Mann-Whitney + AUC + Holm, control por tamaño (estratos y Spearman parcial). Por defecto solo entrenamiento | Sí |
| 5b / 6b | `... 05_estadistica.py --rango 8 16` y `... 06_puntaje.py --rango 8 16` | Lo mismo, solo con nódulos de tamaño parecido (8–16 mm) | Sí |
| 6 | `python scripts/06_puntaje.py` | Puntaje que combina las mejores medidas: diámetro solo vs interior vs interior + entorno, con validación cruzada por paciente y curvas ROC. `--evaluar` usa el 20 % de evaluación (una sola vez, al final) | Sí |
| 6c | `... 05_estadistica.py --sin-calcio` y `... 06_puntaje.py --sin-calcio` | Solo nódulos sin calcio: la pregunta clínica abierta (el calcio ya lo usan los radiólogos) | Sí |
| 7 | `python scripts/07_baja_dosis.py` | ¿Los biomarcadores sobreviven a la baja dosis (tamizaje) y a la compresión? Re-adquiere cada corte en una tomografía simulada, corre todo el pipeline y mide segmentación, reproducibilidad (CCC) y AUC re-armado vs transferido. Lento (~15-30 min); después `--solo-analisis` | Sí |
| 8 (plus) | `python scripts/08_propuesta.py` | Propuesta: cuánto ruido aguanta cada medida (CCC e ICC), un puntaje que solo usa medidas robustas y que separan, y un calcio robusto al ruido (mediana + etiquetado + área mínima). Lee los resultados del 07, no cambia nada del pipeline | Sí |

## Comparaciones (independientes, en cualquier orden salvo la primera)

| Script | Pregunta | Tiempo |
|---|---|---|
| `python comparaciones/reconstruccion.py --casos 3` | ¿Qué método de reconstrucción conviene con baja dosis o pocos ángulos? (SBP, FBP rampa/Hann, SART). **Guarda las reconstrucciones** para los dos de abajo | lento (~30 s por caso) |
| `python comparaciones/filtros.py --casos 3` | ¿Qué filtro limpia mejor la baja dosis? Mediana, gaussiano y Wiener en varios tamaños: ruido que queda vs nitidez del borde | segundos |
| `python comparaciones/robustez.py --casos 3` | ¿Cuánto cambian la intensidad y la textura del nódulo con cada reconstrucción, filtro y JPEG? | segundos |
| `python comparaciones/compresion.py` | ¿Cómo guardar o enviar? Huffman, RLE, PNG, JPEG: tamaño y pérdida | segundos |

Resultados en `resultados/comparaciones/` y `figuras/comparaciones/`.

**Decisiones frente a la materia:** `DECISIONES.md` (qué es de la materia, qué hicimos distinto y por qué).

**Recorrido visual:** `notebooks/recorrido_completo.ipynb` muestra cada paso con imágenes sobre un nódulo (abrir en VS Code y ejecutar celda por celda).

**Probar sin imágenes reales** (con el fantoma): agregar `--carpeta tests/salida/cortes` a cualquier script de `comparaciones/`, y para el pipeline:
```bash
python scripts/04_caracteristicas.py --carpeta tests/salida/cortes --meta tests/salida/meta.csv
```

Los resultados quedan en `resultados/` (CSV) y `figuras/` (PNG). Las imágenes DICOM, los cortes `.npz` y las reconstrucciones guardadas no se suben al repo.

## Estructura

```
tpi/                    módulos (cada uno explica en su docstring el porqué de cada decisión)
  config.py             todos los parámetros en un solo lugar
  compat.py             compatibilidad pylidc/numpy y ~/.pylidcrc
  datos.py              selección de nódulos, división por paciente, corte de trabajo
  segmentacion.py       crecimiento de regiones desde semilla (+ Otsu como comparación)
  validacion.py         Dice, Jaccard, concordancia entre radiólogos, aceptabilidad
  caracteristicas.py    tamaño, forma, espículas (apertura), borde (Sobel), intensidad, composición, GLCM
  entorno.py            pulmón, anillos 0–3 / 3–6 mm, estructuras convergentes (esqueleto)
  mejoramiento.py       mediana, gaussiano, Wiener
  reconstruccion.py     HU→μ, Radon, ruido Poisson, SBP, FBP (rampa/Hann), SART
  compresion.py         ventana a 8 bits, Huffman, RLE, PNG, JPEG, RMSE/SSIM
  estadistica.py        Mann-Whitney, AUC, Holm, estratos por tamaño, Spearman parcial
  puntaje.py            puntaje combinado, validación cruzada por paciente, ROC
  visualizacion.py      figuras
scripts/                pipeline principal, 01 … 06
comparaciones/          una pregunta de "¿qué método conviene?" por archivo
tests/test_fantoma.py   fantoma sintético para verificar el código
```

## Dataset

LIDC-IDRI (TCIA, CC BY 3.0). Armato SG 3rd et al., *Medical Physics* 38:915–931, 2011. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
