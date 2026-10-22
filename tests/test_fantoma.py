"""Fantoma sintético para probar TODO el pipeline sin descargar LIDC.

Genera cortes de tórax falsos (cuerpo, dos pulmones, vasos, ruido) con un
nódulo "benigno" (liso, a veces calcificado) o "sospechoso" (lobulado y
espiculado), más 4 "radiólogos" que dibujan el contorno con pequeñas
diferencias. Sirve para verificar que el código hace lo que creemos: por
ejemplo, la circularidad tiene que dar más baja en los espiculados.

Uso:  python tests/test_fantoma.py
      python scripts/05_caracteristicas.py --carpeta tests/salida/cortes --meta tests/salida/meta.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from tpi.caracteristicas import caracteristicas_interior  # noqa: E402
from tpi.datos import guardar_corte, preparar_corte  # noqa: E402
from tpi.entorno import caracteristicas_entorno, mascara_pulmon  # noqa: E402
from tpi.segmentacion import crecimiento_regiones  # noqa: E402
from tpi.validacion import evaluar  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "salida"
N, PX = 512, 0.7


def forma_nodulo(cr, cc, radio_px, sospechoso, rng):
    rr, cc_ = np.mgrid[:N, :N]
    ang = np.arctan2(rr - cr, cc_ - cc)
    dist = np.hypot(rr - cr, cc_ - cc)
    if sospechoso:
        borde = radio_px * (1 + 0.18 * np.sin(3 * ang + rng.uniform(0, 6)) + 0.08 * np.sin(7 * ang))
    else:
        borde = radio_px * (1 + 0.04 * np.sin(2 * ang + rng.uniform(0, 6)))
    m = dist <= borde
    if sospechoso:  # espículas: líneas finas radiales
        for a in rng.uniform(0, 2 * np.pi, rng.integers(5, 9)):
            largo = radio_px * rng.uniform(1.4, 2.0)
            t = np.linspace(0, largo, 200)
            r = np.clip((cr + t * np.sin(a)).astype(int), 0, N - 1)
            c = np.clip((cc + t * np.cos(a)).astype(int), 0, N - 1)
            m[r, c] = True
    return m


def fantoma(sospechoso, rng):
    rr, cc = np.mgrid[:N, :N]
    img = np.full((N, N), -1000.0)
    cuerpo = ((rr - 256) / 200) ** 2 + ((cc - 256) / 240) ** 2 <= 1
    img[cuerpo] = 40
    pulmones = []
    for c0 in (160, 352):
        p = ((rr - 250) / 160) ** 2 + ((cc - c0) / 80) ** 2 <= 1
        pulmones.append(p)
        img[p] = -850
    # vasos
    for _ in range(25):
        lado = pulmones[rng.integers(2)]
        r0, c0 = np.argwhere(lado)[rng.integers(lado.sum())]
        img[(rr - r0) ** 2 + (cc - c0) ** 2 <= rng.uniform(1, 4) ** 2] = -50
    # nódulo dentro del pulmón izquierdo de la imagen
    radio_mm = rng.uniform(4, 11) if not sospechoso else rng.uniform(6, 14)
    cr, cc0 = 250 + rng.integers(-60, 60), 160 + rng.integers(-25, 25)
    m = forma_nodulo(cr, cc0, radio_mm / PX, sospechoso, rng)
    img[m] = rng.uniform(-30, 50) + (rng.normal(0, 40, m.sum()) if sospechoso else 0)
    if not sospechoso and rng.random() < 0.5:  # calcificación central
        img[(rr - cr) ** 2 + (cc - cc0) ** 2 <= (radio_mm / PX / 3) ** 2] = 600
    if sospechoso and rng.random() < 0.5:  # zona de aire
        img[(rr - cr - 3) ** 2 + (cc - cc0 + 2) ** 2 <= (radio_mm / PX / 4) ** 2] = -800
    img = ndi.gaussian_filter(img, 1.0) + rng.normal(0, 15, img.shape)
    # "radiólogos": cuerpo del nódulo (sin espículas finas) con variaciones
    cuerpo_nod = ndi.binary_opening(m, iterations=2)
    rads = []
    for _ in range(4):
        op = rng.choice(["nada", "dilatar", "erosionar"])
        r = cuerpo_nod
        if op == "dilatar":
            r = ndi.binary_dilation(r)
        elif op == "erosionar":
            r = ndi.binary_erosion(r)
        rads.append(np.roll(r, rng.integers(-1, 2, size=2), axis=(0, 1)))
    consenso = np.mean(rads, axis=0) >= 0.5
    return img.astype(np.float32), consenso, rads


def main(n_por_grupo=6):
    rng = np.random.default_rng(0)
    (SALIDA / "cortes").mkdir(parents=True, exist_ok=True)
    meta = []
    k = 0
    for grupo in ("benigno", "sospechoso"):
        for _ in range(n_por_grupo):
            img, cons, rads = fantoma(grupo == "sospechoso", rng)
            d = preparar_corte(img, PX, cons, rads)
            nombre = f"FANTOMA-{k:04d}_s{k}_n0"
            guardar_corte(SALIDA / "cortes" / f"{nombre}.npz", d)
            meta.append(dict(paciente=f"FANTOMA-{k:04d}", scan_id=k, nodulo=0, grupo=grupo,
                             conjunto="entrenamiento", malignidad_mediana=5 if grupo == "sospechoso" else 1,
                             diametro_medio_mm=np.nan))
            # chequeos rápidos
            seg = crecimiento_regiones(d["img"], d["semilla"])
            v = evaluar(seg, d["consenso"], d["radiologos"])
            f = caracteristicas_interior(d["img"], seg)
            pul = mascara_pulmon(d["corte_completo_rem"])
            assert pul.sum() > 0
            assert v["dice"] > 0.6, (nombre, v)
            assert 0 < f["circularidad"] < 1.3
            k += 1
            print(f"{nombre} {grupo:10s} Dice {v['dice']:.2f} conc {v['concordancia_radiologos']:.2f} "
                  f"circ {f['circularidad']:.2f} solidez {f['solidez']:.2f} calcio {f['frac_calcio']:.2f}")
    pd.DataFrame(meta).to_csv(SALIDA / "meta.csv", index=False)
    print("OK fantoma ->", SALIDA)


if __name__ == "__main__":
    main()
