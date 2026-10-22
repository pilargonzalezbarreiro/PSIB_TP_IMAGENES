"""Métricas de solapamiento y umbral de aceptabilidad.

Dice = 2|A∩B| / (|A|+|B|);  Jaccard = |A∩B| / |A∪B|  (J = D / (2 - D)).

Idea central del anteproyecto: no fijamos un Dice "bueno" arbitrario
(p. ej. 0,8), sino que le pedimos al algoritmo lo mismo que se le pide a un
radiólogo. Si los radiólogos entre sí coinciden con Dice medio 0,75 en un
nódulo, una segmentación con Dice ≥ 0,75 contra el consenso es tan buena
como la de un experto más.
"""
from itertools import combinations

import numpy as np


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u


def concordancia_radiologos(mascaras):
    """Media de los Dice entre todos los pares de radiólogos en el corte.
    Se ignoran radiólogos sin contorno en este corte (máscara vacía)."""
    validas = [m for m in mascaras if m.any()]
    if len(validas) < 2:
        return np.nan
    return float(np.mean([dice(a, b) for a, b in combinations(validas, 2)]))


def evaluar(mascara_alg, consenso, mascaras_radiologos):
    d = dice(mascara_alg, consenso)
    conc = concordancia_radiologos(mascaras_radiologos)
    return dict(
        dice=d, jaccard=jaccard(mascara_alg, consenso),
        concordancia_radiologos=conc,
        aceptable=bool(np.isfinite(conc) and d >= conc),
    )
