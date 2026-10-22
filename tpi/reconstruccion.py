"""Simulación de la adquisición tomográfica y reconstrucción.

Cadena física:
  HU -> μ (coeficiente de atenuación)       μ = μ_agua (1 + HU/1000)
  μ  -> proyecciones p(θ, s) = ∫ μ dl         transformada de Radon
  p  -> cuentas de fotones                    I = I0 e^{-p}  (Beer-Lambert)
        con ruido Poisson: menos fotones (I0 chico = baja dosis) -> más ruido
  cuentas -> p̂ = -ln(I/I0) -> reconstrucción (FBP o SART) -> HU

FBP (retroproyección filtrada): rápida, analítica; el filtro rampa realza
altas frecuencias, por eso amplifica el ruido.
SART: iterativa (resuelve Ax = b de a una proyección por vez); con pocas
iteraciones regulariza implícitamente y tolera mejor pocas proyecciones.
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


def reconstruir(p, pixel_mm, metodo="fbp", iteraciones=2):
    th = angulos(p.shape[1])
    p = p / pixel_mm  # volver a unidades por píxel
    if metodo == "fbp":
        mu = iradon(p, theta=th, filter_name="ramp", circle=True)
    elif metodo == "sart":
        mu = None
        for _ in range(iteraciones):
            mu = iradon_sart(p, theta=th, image=mu)
    else:
        raise ValueError(metodo)
    return mu_a_hu(mu)


def simular(corte_hu, pixel_mm, n_angulos=C.N_ANGULOS_COMPLETO, I0=None,
            metodo="fbp", iteraciones=2, rng=None):
    """Corte en HU -> corte reconstruido en HU bajo una condición de adquisición.
    I0=None: proyecciones sin ruido (solo error de discretización)."""
    hu = np.maximum(corte_hu, -1000)
    p = sinograma(hu_a_mu(hu), pixel_mm, n_angulos)
    if I0 is not None:
        p = agregar_ruido(p, I0, rng)
    rec = reconstruir(p, pixel_mm, metodo, iteraciones)
    return rec.astype(np.float32), p


def mascara_fov(n):
    rr, cc = np.ogrid[:n, :n]
    return (rr - (n - 1) / 2) ** 2 + (cc - (n - 1) / 2) ** 2 <= (n / 2 - 1) ** 2
