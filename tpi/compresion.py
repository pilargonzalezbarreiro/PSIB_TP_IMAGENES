"""Compresión y descompresión (sección 7 de la materia).

Paso previo: la tomografía tiene ~12 bits por píxel (HU de -1024 a ~3000),
y los formatos de imagen comunes (y JPEG baseline) trabajan con 8 bits. Se
aplica la ventana de pulmón [-1000, 400] HU y se lleva a 0-255: cada nivel
≈ 5,5 HU. Esa cuantización ya pierde algo (error ≈ 5,5/√12 ≈ 1,6 HU); se
reporta aparte para no confundirla con la pérdida del compresor.

Métodos, uno por cada redundancia que vimos:
  Huffman  (redundancia de código, sin pérdida): códigos cortos para los
           tonos frecuentes. No le importa dónde están los píxeles.
  RLE      (redundancia de píxeles, sin pérdida): "valor + cuántas veces
           seguidas". Rinde con zonas uniformes (el aire de afuera del
           cuerpo), empeora con ruido (el interior del tórax).
  PNG      (sin pérdida): DEFLATE = LZ77 (diccionario de secuencias
           repetidas, de la familia de LZW) + Huffman. Combina las dos ideas.
  JPEG     (redundancia visual, CON pérdida): bloques 8x8 -> DCT ->
           cuantización (acá se pierde, sobre todo altas frecuencias: bordes
           y texturas finas) -> zigzag + RLE + Huffman.

Huffman y RLE se implementan acá solo para CALCULAR cuántos bits ocuparía
la imagen (no se escribe un archivo); no se cuenta la tabla de códigos.

Métricas de calidad:
  RMSE (en HU): error promedio, en las unidades de la imagen.
  SSIM (0 a 1): compara estructura local (bordes, contraste). Se suma a RMSE
  porque RMSE no distingue entre "ruido" y "borroneo": una imagen
  suavizada puede tener RMSE bajo y haber perdido los bordes, y eso es
  justamente lo que nos importa para la textura.
  PSNR (dB): el mismo error que RMSE en escala logarítmica, relativo al
  rango de la imagen (20·log10(rango/RMSE)); se reporta porque es la métrica
  estándar para comparar compresiones (y la que figura en el anteproyecto).
"""
import heapq
import io
from collections import Counter

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from . import config as C


def hu_a_8bits(hu, ventana=C.VENTANA_8BIT):
    lo, hi = ventana
    return np.round((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def de_8bits_a_hu(img8, ventana=C.VENTANA_8BIT):
    lo, hi = ventana
    return img8.astype(np.float32) / 255 * (hi - lo) + lo


def comprimir(img8, formato="JPEG", calidad=75):
    """Devuelve (imagen descomprimida uint8, bytes del archivo)."""
    buf = io.BytesIO()
    kw = {"quality": int(calidad)} if formato == "JPEG" else {"optimize": True}
    Image.fromarray(img8).save(buf, format=formato, **kw)
    n = buf.tell()
    buf.seek(0)
    return np.array(Image.open(buf)), n


def largos_huffman(frecuencias):
    """Árbol de Huffman como en clase: unir siempre los dos menos frecuentes.
    Devuelve {símbolo: largo del código en bits}."""
    if len(frecuencias) == 1:
        return {next(iter(frecuencias)): 1}
    heap = [(f, i, {s: 0}) for i, (s, f) in enumerate(frecuencias.items())]
    heapq.heapify(heap)
    k = len(heap)
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        unidos = {s: l + 1 for s, l in {**a, **b}.items()}
        heapq.heappush(heap, (f1 + f2, k, unidos))
        k += 1
    return heap[0][2]


def bytes_huffman(img8):
    frec = Counter(img8.ravel().tolist())
    largos = largos_huffman(frec)
    return sum(frec[s] * largos[s] for s in frec) / 8


def bytes_rle(img8):
    """RLE con pares (valor: 1 byte, repeticiones: 1 byte, máx. 255).
    Se prueba recorrido horizontal y vertical y se queda con el mejor."""
    def contar(x):
        x = x.ravel()
        cortes = np.flatnonzero(np.diff(x)) + 1
        largos = np.diff(np.concatenate([[0], cortes, [x.size]]))
        return int(np.ceil(largos / 255).sum()) * 2
    return min(contar(img8), contar(img8.T))


def metricas(referencia, prueba, rango, mascara=None):
    ref = np.asarray(referencia, float)
    pru = np.asarray(prueba, float)
    err = (ref - pru)[mascara] if mascara is not None else (ref - pru)
    if mascara is not None:
        ref, pru = np.where(mascara, ref, 0), np.where(mascara, pru, 0)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return dict(rmse=rmse,
                psnr=float(20 * np.log10(rango / rmse)) if rmse > 0 else np.inf,
                ssim=float(structural_similarity(ref, pru, data_range=rango)))
