# Decisiones del pipeline frente a la materia

Criterio general:
1. **La base es la materia.** No reemplazamos un método de la materia por uno equivalente de afuera (por ejemplo, no usamos "disimilitud" en lugar de contraste).
2. **Construimos encima.** Donde el problema pide más (clasificar nódulos), agregamos medidas y pasos nuevos, preferentemente armados con herramientas de la materia usadas de otra forma (lo que borra la apertura, el esqueleto del entorno, el gradiente del borde).
3. **Si el problema real obliga a hacer algo distinto** (nódulos chicos, distinto píxel, HU), se hace y se justifica acá.
4. **Lo que agregamos no se da por bueno de antemano:** la estadística (05) y el puntaje (06) dicen si sirvió.

## 1. Igual que en la materia

| Etapa | Qué usamos | Sección del resumen |
|---|---|---|
| Análisis | Mirar histogramas y tipo de ruido antes de elegir métodos | 2 |
| Reconstrucción | Beer-Lambert, Radon, sinograma, SBP, FBP con rampa (Ram-Lak) y con Hann, SART | 1.3–1.10 |
| Mejoramiento | Mediana (impulsivo), gaussiano (granulado) y Wiener (adaptativo), elegidos según el tipo de ruido | 3 |
| Segmentación | Crecimiento de regiones desde semilla (conectividad 8); Otsu como comparación | 4.3, 4.5, 4.10 |
| Posprocesamiento | Apertura (cortar vasos pegados), etiquetado, eliminar objetos chicos por área, cierre | 5.2–5.5 |
| Evaluación | Dice y Jaccard | 4.11 |
| Características | Área, intensidad media y desvío (máscara × original), GLCM | 6.6, 6.7 |
| GLCM | 16 niveles, distancia 1, 4 direcciones promediadas, simetrizada y normalizada, solo dentro de la región; energía, entropía, contraste, homogeneidad, correlación, cluster shade y prominence | 6.2–6.4 |
| Entorno | Anillos por dilatación ("margen alrededor de un tumor"), calculada con la transformada de distancia (dilatación con disco euclídeo exacto; el disco de píxeles cambiaba los anillos un 11–17 %) | 5.3 |
| Compresión | Huffman, RLE, PNG (sin pérdida), JPEG por calidades (con pérdida) | 7 |

## 2. Construido encima de la materia

| Qué agregamos | Herramienta de la materia en la que se apoya | Para qué |
|---|---|---|
| **Circularidad, solidez, excentricidad** | Propiedades de la región (área, perímetro, envolvente, momentos) | La materia mide área y bounding box; el signo de malignidad es el **borde** (lobulado, espiculado) y la forma. Excentricidad mide alargamiento, algo que la circularidad no ve (un óvalo liso) |
| **Fracción fina** (`frac_fina`) | Apertura: "elimina estructuras delgadas" | En vez de usar la apertura solo para limpiar, se mide **lo que borra**: la parte del nódulo de menos de ~3 mm de ancho = espículas |
| **Nitidez del borde** (HU/mm) | Detección de bordes (discontinuidad), gradiente de Sobel | Versión objetiva del "margen" que califican los radiólogos: borde nítido vs difuso |
| **Estructuras convergentes** (cuántas llegan al nódulo, longitud por mm de borde) | Esqueletización ("vasos: esqueleto → largo, ramificaciones") + etiquetado | Signo de convergencia vascular y espículas que se extienden: el nódulo maligno tracciona el tejido |
| **Forma con y sin apertura** (`*_cruda`) | Apertura | La apertura corta vasos pero también afeita espículas; medir las dos versiones muestra cuánta información se pierde |
| **Asimetría y curtosis** de las HU | Histograma de la región | Forma del histograma (colas por necrosis, calcio, burbujas) que la media y el desvío no capturan. Riesgo: con ~60 píxeles pueden ser ruidosas |
| **Fracción de calcio y de aire** | Umbral sobre la región | Específicas de TC: calcio → suele ser benigno; aire/burbujas → puede ser sospechoso |
| **Puntaje combinado** (script 06) | "Vector de características para clasificación" | Combina las k medidas más discriminativas (no repetidas) en un número; se compara diámetro solo vs interior vs interior + entorno con validación cruzada por paciente y curvas ROC |
| **Biomarcadores bajo baja dosis y compresión** (script 07) | Reconstrucción (Radon, Beer-Lambert, ruido de Poisson, FBP) + compresión JPEG + todo el pipeline | El tamizaje se hace con baja dosis. Cada corte real se re-adquiere en una tomografía simulada con menos fotones (y se comprime), se corre el pipeline completo y se mide: segmentación, reproducibilidad de cada medida (CCC de Lin) y AUC del puntaje **re-armado** con esa dosis y **transferido** (armado con dosis normal, aplicado a baja dosis: lo que pasa cuando un modelo publicado se usa en tamizaje) |
| **Robustez** (`comparaciones/robustez.py`) | "El filtrado cambia la textura; JPEG altera la GLCM" | Se cuantifica: misma máscara, medidas sobre cada reconstrucción, filtro y JPEG |
| **Evaluación de filtros** (`comparaciones/filtros.py`) | Elegir filtro según el ruido | Además de RMSE/PSNR/SSIM: ruido residual en el parénquima y nitidez del borde del nódulo, con varios tamaños de cada filtro |

## 3. Distinto de la materia, porque el problema lo pide

| Decisión | Qué dice la materia | Qué hicimos y por qué |
|---|---|---|
| Nódulos pegados a la pared | Region growing "se escapa con bordes débiles"; "de lo grande a lo chico" (4.10) | Con datos reales, ~30 % de los nódulos (yuxtapleurales) se escapaban por toda la pared del tórax (Dice < 0,3, región de 40–48 mm). **Se segmenta primero el pulmón y se le aplica un cierre con disco grande (rolling ball)** que rellena la muesca del nódulo sin entrar en la pared; el nódulo solo crece adentro. El radio del cierre (10/15/20/30 mm) se ajusta en entrenamiento junto con T. El cierre se calcula con la transformada de distancia (mismo resultado, disco euclídeo exacto, instantáneo) |
| Criterio del crecimiento de regiones | "Diferencia con la semilla ≤ tolerancia" | **HU > T.** No depende del valor de un único píxel (la semilla puede caer sobre calcio o ruido). Es "parecido al tejido" con tolerancia fija. T se ajusta con el conjunto de entrenamiento |
| Huecos internos | Cierre | **Relleno de huecos.** El cierre también suaviza el borde y tapa los entrantes entre lóbulos, que es justo lo que mide la forma. Las cavidades internas son parte del nódulo |
| Rango de cuantización de la GLCM | Reducir a 8–16 niveles | 16 niveles, pero **repartidos en [−500, 300] HU para el nódulo y [−1000, 0] HU para el entorno**. Con [−1000, 400], cada nivel vale 87 HU y el interior del nódulo cae en 2–3 niveles. El rango es fijo e igual para todos los nódulos (condición de la materia) |
| Iteraciones de SART | "Suele bastar una" | Con 180 vistas, sí. **Con 60 vistas, una iteración subestima las HU** (no convergió): se muestran 1 y 3 iteraciones |
| ART | Método de la materia | **No implementado.** Requiere armar A (rayos × píxeles): ~92.000 × 262.144 para 512×512. SART resuelve el mismo sistema sin armar A |
| Perímetro para circularidad y solidez | Área y perímetro de la región | Contar píxeles de borde subestima el perímetro de objetos chicos: un círculo perfecto de 5 mm daba circularidad 1,25 y uno de 20 mm 0,91, y en datos reales los benignos (chicos) salían con circularidad > 1. **Área y perímetro del mismo contorno sub-píxel** (isolínea 0,5 de la máscara apenas suavizada): el círculo da 0,96–0,99 en todos los tamaños y la circularidad no puede pasar de 1 |
| Calcio como confundidor | — (no está en la materia) | Con 217 nódulos, el 57 % de los benignos tiene calcio (> 200 HU) contra el 17 % de los sospechosos, y las medidas que "separan más allá del tamaño" (contraste GLCM, asimetría, desvío, nitidez del borde) están arrastradas por el calcio. Pero el calcio es justamente lo que el radiólogo ya mira (y Lung-RADS ya clasifica como benigno): encontrarlo no es un hallazgo. **Se agrega el análisis `--sin-calcio` (05 y 06)**, que responde la pregunta clínica abierta: en nódulos sólidos SIN calcio, ¿algo separa más que el diámetro? |
| "Entorno solo" en el puntaje | — | El puntaje toma las k mejores medidas; si las del entorno son más débiles, "interior + entorno" sale idéntico a "interior" y no se sabe si el entorno tiene señal. Se agrega un puntaje solo con el entorno |
| Radio del cierre del pulmón | — | Con 20/30/40 mm el Dice bajaba al crecer el radio y el mejor quedó en el borde de la grilla: se prueba 10/15/20/30 mm. Las fugas que quedan son de nódulos pegados al mediastino o al hilio: un disco grande rellena también esa concavidad |
| Control por tamaño | Estratos de diámetro (anteproyecto) | En LIDC los benignos son casi todos < 10 mm y los sospechosos > 10 mm: con la muestra original casi no hay nódulos de tamaño parecido en los dos grupos y los estratos quedan vacíos. **Se amplía la muestra con todos los nódulos elegibles de 8 a 16 mm** (81 benignos y 75 sospechosos en total) y se compara dentro de ese rango (`--rango 8 16` en 05 y 06), con el diámetro de los radiólogos para no depender de la segmentación |
| Ecualización | Para bajo contraste | **No se usa.** Cambia los valores de cada imagen de forma distinta y las HU dejarían de ser comparables. En su lugar, ventana de HU fija para visualizar y para pasar a 8 bits |
| Filtrado antes de medir | "Si no hace falta, no filtres" | **No se filtra antes de segmentar ni de medir.** El ruido (~15–30 HU) es chico frente al salto pulmón–tejido (~850 HU) |
| Organización | — | **Pipeline (`scripts/`) separado de comparaciones (`comparaciones/`).** El pipeline mide sobre la imagen original; las comparaciones responden "¿qué método conviene?" sin alimentar los resultados |

## 4. Fuera de la materia (imprescindible para el problema)

| Qué | Por qué |
|---|---|
| Remuestreo a 0,7 mm | Cada tomógrafo tiene otro tamaño de píxel. Sin esto, las medidas en píxeles no son comparables |
| Consenso de radiólogos (≥ 50 %) y umbral de aceptabilidad | Referencia para validar: aceptable si Dice(algoritmo, consenso) ≥ Dice medio entre radiólogos |
| División entrenamiento / evaluación por paciente | Ajustar con unos nódulos y evaluar con otros, sin que el mismo paciente quede en los dos |
| Métricas de calidad de imagen | RMSE (en HU), PSNR (estándar en compresión, figura en el anteproyecto) y SSIM (distingue ruido de borroneo, que RMSE mezcla) |
| Estadística (Mann-Whitney, AUC, Holm, estratos por tamaño, Spearman parcial) | El paso "comparación" con ~100 nódulos y ~40 medidas necesita una prueba que no suponga normalidad, corrección por comparaciones múltiples y dos controles del tamaño (confundidor), como promete el anteproyecto |
| Validación cruzada por paciente y bootstrap | Para evaluar el puntaje con nódulos que no se usaron para armarlo, y dar un intervalo de confianza honesto con pocos casos |
