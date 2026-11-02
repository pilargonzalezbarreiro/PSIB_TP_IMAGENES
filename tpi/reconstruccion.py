"""Simulación de la adquisición tomográfica y reconstrucción (sección 1 de
la materia).

Problema directo (adquisición) — lo que haría el tomógrafo:
  HU -> μ                          μ = μ_agua (1 + HU/1000)  (la imagen que
                                   queremos es el mapa de μ)
  μ  -> sinograma p(θ, s)          transformada de Radon (suma de μ por rayo)
  p  -> fotones detectados         Beer-Lambert: I = I0 e^{-p}
        con ruido de Poisson: cada detector cuenta fotones, y contar tiene
        un error ~ √cuentas. Menos dosis (I0 chico) = menos fotones = más ruido
        relativo (crece como 1/√I0). Es el "granulado por ruido en la
        adquisición" de la tabla de artefactos.
  fotones -> p̂ = -ln(I/I0)         (el logaritmo vuelve lineal el problema)

Problema inverso (reconstrucción), métodos de la materia:
  SBP   retroproyección simple: "unta" cada proyección; borrosa con halo
        (sobre-pondera bajas frecuencias). Solo didáctica.
  FBP   filtro en frecuencia y después retroproyección. Rampa pura
        (Ram-Lak): nítida pero amplifica el ruido -> para datos limpios.
        Hann: rampa con ventana que frena las frecuencias altísimas -> para
        datos ruidosos (baja dosis). Compromiso resolución-ruido.
  SART  iterativa, corrige una vista entera por vez promediando los
        residuos -> tolera pocas proyecciones y ruido. La materia: "suele
        bastar una iteración; más iteraciones -> más ruido".

ART no se incluye: necesita armar la matriz A (una fila por rayo, una
columna por píxel). Para un corte de 512 x 512 con 180 ángulos son ~92.000
rayos x 262.144 píxeles: inviable fuera de un ejemplo de juguete. SART
resuelve el mismo sistema A·x = b sin armar A explícitamente.
"""
import numpy as np
from skimage.transform import iradon, iradon_sart, radon

from . import config as C


def hu_a_mu(hu):
    return np.maximum(C.MU_AGUA * (1 + np.asarray(hu, float) / 1000.0), 0.0)


def mu_a_hu(mu):
    return 1000.0 * (mu / C.MU_AGUA - 1.0)


def angulos(n):
    return np.linspace(0.0, 180.0, n, endpoint=False)


def _a_circulo(img):
    """radon(circle=True) asume que todo fuera del círculo inscripto es 0."""
    n = img.shape[0]
    rr, cc = np.ogrid[:n, :n]
    fuera = (rr - (n - 1) / 2) ** 2 + (cc - (n - 1) / 2) ** 2 > (n // 2 - 1) ** 2
    img = img.copy()
    img[fuera] = 0
    return img


def sinograma(mu, pixel_mm, n_angulos):
    """Integrales de línea en unidades adimensionales (μ en mm^-1 x mm)."""
    return radon(_a_circulo(mu), theta=angulos(n_angulos), circle=True) * pixel_mm


def agregar_ruido(p, I0, rng=None):
    rng = np.random.default_rng(C.SEMILLA) if rng is None else rng
    cuentas = rng.poisson(I0 * np.exp(-p))
    return -np.log(np.maximum(cuentas, 1) / I0)


def reconstruir(p, pixel_mm, metodo="fbp", iteraciones=1, filtro="ramp"):
    """metodo: 'sbp', 'fbp' (filtro 'ramp' = Ram-Lak o 'hann') o 'sart'.
    Devuelve HU. La SBP no está bien escalada (no es la inversa de Radon,
    es su adjunto): sus valores solo sirven para mirar la imagen."""
    th = angulos(p.shape[1])
    p = p / pixel_mm  # volver a unidades por píxel
    if metodo == "sbp":
        mu = iradon(p, theta=th, filter_name=None, circle=True)
    elif metodo == "fbp":
        mu = iradon(p, theta=th, filter_name=filtro, circle=True)
    elif metodo == "sart":
        mu = None
        for _ in range(iteraciones):
            mu = iradon_sart(p, theta=th, image=mu)
    else:
        raise ValueError(metodo)
    return mu_a_hu(mu)


def simular(corte_hu, pixel_mm, n_angulos=C.N_ANGULOS_COMPLETO, I0=None,
            metodo="fbp", iteraciones=1, filtro="ramp", rng=None):
    """Corte en HU -> corte reconstruido en HU bajo una condición de adquisición.
    I0=None: proyecciones sin ruido (solo error de discretización)."""
    hu = np.maximum(corte_hu, -1000)
    p = sinograma(hu_a_mu(hu), pixel_mm, n_angulos)
    if I0 is not None:
        p = agregar_ruido(p, I0, rng)
    rec = reconstruir(p, pixel_mm, metodo, iteraciones, filtro)
    return rec.astype(np.float32), p


def mascara_fov(n):
    rr, cc = np.ogrid[:n, :n]
    return (rr - (n - 1) / 2) ** 2 + (cc - (n - 1) / 2) ** 2 <= (n / 2 - 1) ** 2
