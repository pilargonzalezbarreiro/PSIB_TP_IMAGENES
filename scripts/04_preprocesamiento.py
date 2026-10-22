"""Etapa 4 - Pre-procesamiento (demostración sobre algunos casos):
reconstrucción tomográfica simulada, mejoramiento con filtros y compresión.

Se trabaja sobre el CORTE COMPLETO con su tamaño de píxel original (la
reconstrucción necesita todo el campo de visión).

Salidas:
  resultados/04_reconstruccion.csv   RMSE/PSNR/SSIM por condición
  resultados/04_filtros.csv          ganancia de cada filtro sobre baja dosis
  resultados/04_compresion.csv       tasa de compresión y calidad
  figuras/04_*.png

Uso:  python scripts/04_preprocesamiento.py [--casos 3] [--carpeta datos/cortes]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import compresion as Z  # noqa: E402
from tpi import config as C  # noqa: E402
from tpi import reconstruccion as R  # noqa: E402
from tpi.datos import cargar_corte  # noqa: E402
from tpi.mejoramiento import FILTROS  # noqa: E402
from tpi.visualizacion import figura_grilla  # noqa: E402

RANGO_HU = 1400.0  # rango de la ventana de pulmón [-1000, 400]


def recorte_nodulo(img, d, lado_mm=64):
    """Recorte alrededor del nódulo en el corte completo (resolución original)."""
    esc = C.PIXEL_MM / float(d["pixel_mm_original"])
    r, c = (np.asarray(d["semilla_completo"]) * esc).astype(int)
    h = int(lado_mm / float(d["pixel_mm_original"]) / 2)
    return img[max(r - h, 0):r + h, max(c - h, 0):c + h]


def condiciones():
    yield "FBP 180 ang, sin ruido", dict(n_angulos=180, I0=None, metodo="fbp")
    yield "FBP 180 ang, I0=100000 (ref)", dict(n_angulos=180, I0=C.I0_REFERENCIA, metodo="fbp")
    for I0 in C.I0_BAJA_DOSIS:
        yield f"FBP 180 ang, I0={I0:g}", dict(n_angulos=180, I0=I0, metodo="fbp")
    yield "FBP 60 ang, I0=100000", dict(n_angulos=60, I0=C.I0_REFERENCIA, metodo="fbp")
    yield "SART 60 ang (2 it), I0=100000", dict(n_angulos=60, I0=C.I0_REFERENCIA, metodo="sart")
    yield f"SART 180 ang (2 it), I0={C.I0_BAJA_DOSIS[-1]:g}", dict(
        n_angulos=180, I0=C.I0_BAJA_DOSIS[-1], metodo="sart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", type=int, default=3)
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    args = ap.parse_args()
    archivos = sorted(Path(args.carpeta).glob("*.npz"))[:args.casos]
    if not archivos:
        sys.exit("No hay cortes. Corré antes 03_extraer_cortes.py (o tests/test_fantoma.py).")
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    filas_rec, filas_fil, filas_comp = [], [], []

    for n, ruta in enumerate(archivos):
        d = cargar_corte(ruta)
        orig = np.maximum(d["corte_completo"], -1000)
        px = float(d["pixel_mm_original"])
        fov = R.mascara_fov(orig.shape[0])
        rng = np.random.default_rng(C.SEMILLA)
        print(f"[{ruta.stem}] {orig.shape}, píxel {px:.3f} mm")

        # --- Reconstrucción ---
        recs = {}
        for nombre, kw in condiciones():
            rec, _ = R.simular(orig, px, rng=rng, **kw)
            recs[nombre] = rec
            m = Z.metricas(np.clip(orig, -1000, 400), np.clip(rec, -1000, 400), RANGO_HU, fov)
            # Ruido en HU dentro del pulmón/aire uniforme: desvío del error
            filas_rec.append(dict(caso=ruta.stem, condicion=nombre, **m))

        # --- Mejoramiento: filtrar la reconstrucción más ruidosa ---
        ruidosa = recs[f"FBP 180 ang, I0={C.I0_BAJA_DOSIS[-1]:g}"]
        filtradas = {"sin filtro": ruidosa}
        for nf, f in FILTROS.items():
            filtradas[nf] = f(ruidosa)
        for nf, im in filtradas.items():
            m = Z.metricas(np.clip(orig, -1000, 400), np.clip(im, -1000, 400), RANGO_HU, fov)
            filas_fil.append(dict(caso=ruta.stem, filtro=nf, **m))

        # --- Compresión ---
        img8 = Z.hu_a_8bits(orig)
        crudo = img8.size  # bytes sin comprimir a 8 bits
        hu8 = Z.de_8bits_a_hu(img8)
        m = Z.metricas(np.clip(orig, -1000, 400), hu8, RANGO_HU)
        filas_comp.append(dict(caso=ruta.stem, formato="8 bits sin comprimir (ventana)",
                               bytes=crudo, tasa=1.0, **m))
        png, nb = Z.comprimir(img8, "PNG")
        assert np.array_equal(png, img8), "PNG debería ser sin pérdida"
        filas_comp.append(dict(caso=ruta.stem, formato="PNG", bytes=nb, tasa=crudo / nb, **m))
        jpgs = {}
        for q in C.CALIDADES_JPEG:
            jpg, nb = Z.comprimir(img8, "JPEG", q)
            jpgs[q] = Z.de_8bits_a_hu(jpg)
            mj = Z.metricas(np.clip(orig, -1000, 400), jpgs[q], RANGO_HU)
            m8 = Z.metricas(img8, jpg, 255)
            filas_comp.append(dict(caso=ruta.stem, formato=f"JPEG q{q}", bytes=nb, tasa=crudo / nb,
                                   psnr_vs_8bits=m8["psnr"], **mj))

        if n == 0:  # figuras del primer caso
            figura_grilla([orig] + list(recs.values()), ["original"] + list(recs),
                          C.DIR_FIGURAS / "04_reconstruccion_completo.png",
                          suptitulo="Reconstrucción simulada (corte completo)")
            figura_grilla([recorte_nodulo(orig, d)] + [recorte_nodulo(v, d) for v in recs.values()],
                          ["original"] + list(recs), C.DIR_FIGURAS / "04_reconstruccion_zoom.png",
                          suptitulo="Reconstrucción simulada (zoom 64 mm al nódulo)")
            figura_grilla([recorte_nodulo(v, d) for v in filtradas.values()], list(filtradas),
                          C.DIR_FIGURAS / "04_filtros_zoom.png",
                          suptitulo=f"Filtros sobre FBP de baja dosis (I0={C.I0_BAJA_DOSIS[-1]:g})")
            figura_grilla([recorte_nodulo(orig, d)] + [recorte_nodulo(v, d) for v in jpgs.values()],
                          ["original"] + [f"JPEG q{q}" for q in jpgs],
                          C.DIR_FIGURAS / "04_compresion_zoom.png", columnas=5,
                          suptitulo="Compresión JPEG (zoom al nódulo)")
            # Mapa de error JPEG q25 para ver los bloques 8x8
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(recorte_nodulo(jpgs[25] - hu8, d), cmap="RdBu", vmin=-60, vmax=60)
            ax.set_title("Error JPEG q25 (HU)")
            ax.axis("off")
            fig.savefig(C.DIR_FIGURAS / "04_error_jpeg_q25.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    C.DIR_RESULTADOS.mkdir(exist_ok=True)
    for nombre, filas, clave in (("04_reconstruccion", filas_rec, "condicion"),
                                 ("04_filtros", filas_fil, "filtro"),
                                 ("04_compresion", filas_comp, "formato")):
        df = pd.DataFrame(filas)
        df.to_csv(C.DIR_RESULTADOS / f"{nombre}.csv", index=False)
        print(f"\n== {nombre} (promedio de {len(archivos)} casos) ==")
        print(df.groupby(clave, sort=False).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
