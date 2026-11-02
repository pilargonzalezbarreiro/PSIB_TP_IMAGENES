"""¿Qué método de reconstrucción conviene según la adquisición?

Simulación en dos pasos (no partimos de proyecciones: LIDC trae imágenes ya
reconstruidas):
  1. problema directo: a la imagen de LIDC se le aplica Radon -> el sinograma
     que "habría medido" el tomógrafo; ahí se le agrega ruido de baja dosis o
     se le sacan ángulos;
  2. problema inverso: se reconstruye ese sinograma con distintos métodos y
     se compara contra la imagen original (que hace de "verdad").

Condiciones (cada una responde una pregunta de la materia):
  SBP vs FBP sin ruido ............ ¿por qué hace falta el filtro? (halo)
  FBP rampa con 3 dosis ........... ¿el ruido crece como 1/√dosis?
  FBP rampa vs Hann vs SART ....... con baja dosis, ¿qué método tolera el ruido?
  FBP vs SART (1 y 3 it) .......... con pocos ángulos, ¿qué método tolera la falta de datos?

Las reconstrucciones se guardan en datos/cache_reconstrucciones/ para que
filtros.py y robustez.py no tengan que recalcularlas (es lo lento: 2-6 s cada una).

Salidas:
  resultados/comparaciones/reconstruccion.csv
  figuras/comparaciones/reconstruccion_*.png, sbp_vs_fbp.png

Uso:  python comparaciones/reconstruccion.py [--casos 3] [--carpeta datos/cortes]
"""
from _comun import (BAJA, DIR_CACHE, DIR_FIG, DIR_RES, RANGO_HU, C, argumentos, cargar_corte,
                    listar_casos, np, original, pd, recorte_nodulo, ruta_cache)

import matplotlib.pyplot as plt  # noqa: E402

from tpi import compresion as Z  # noqa: E402
from tpi import reconstruccion as R  # noqa: E402
from tpi.visualizacion import figura_grilla  # noqa: E402


def condiciones():
    yield "SBP 180, sin ruido", dict(n_angulos=180, metodo="sbp")
    yield "FBP rampa 180, sin ruido", dict(n_angulos=180, metodo="fbp")
    yield "FBP rampa 180, dosis ref", dict(n_angulos=180, I0=C.I0_REFERENCIA, metodo="fbp")
    for I0 in C.I0_BAJA_DOSIS:
        yield f"FBP rampa 180, I0={I0:g}", dict(n_angulos=180, I0=I0, metodo="fbp")
    yield f"FBP Hann 180, I0={BAJA:g}", dict(n_angulos=180, I0=BAJA, metodo="fbp", filtro="hann")
    yield f"SART 180 (1 it), I0={BAJA:g}", dict(n_angulos=180, I0=BAJA, metodo="sart")
    yield "FBP rampa 60, dosis ref", dict(n_angulos=60, I0=C.I0_REFERENCIA, metodo="fbp")
    # La materia: "suele bastar una iteración". Con 180 vistas es cierto; con 60,
    # 1 iteración todavía no convergió (subestima las HU), por eso 1 vs 3.
    yield "SART 60 (1 it), dosis ref", dict(n_angulos=60, I0=C.I0_REFERENCIA, metodo="sart")
    yield "SART 60 (3 it), dosis ref", dict(n_angulos=60, I0=C.I0_REFERENCIA, metodo="sart",
                                            iteraciones=3)


def main():
    args = argumentos(__doc__.splitlines()[0]).parse_args()
    archivos = listar_casos(args.carpeta, args.casos)
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    filas = []
    for n, ruta in enumerate(archivos):
        d = cargar_corte(ruta)
        orig = original(d)
        px = float(d["pixel_mm_original"])
        fov = R.mascara_fov(orig.shape[0])
        rng = np.random.default_rng(C.SEMILLA)
        print(f"[{ruta.stem}] píxel {px:.3f} mm")
        recs = {}
        for nombre, kw in condiciones():
            rec, _ = R.simular(orig, px, rng=rng, **kw)
            recs[nombre] = rec
            if kw["metodo"] == "sbp":  # no está escalada: el error en HU no tiene sentido
                continue
            m = Z.metricas(np.clip(orig, -1000, 400), np.clip(rec, -1000, 400), RANGO_HU, fov)
            filas.append(dict(caso=ruta.stem, condicion=nombre, **m))
            print(f"   {nombre:28s} RMSE {m['rmse']:6.1f} HU  PSNR {m['psnr']:5.1f} dB  SSIM {m['ssim']:.3f}")
        np.savez_compressed(ruta_cache(ruta), **recs)

        if n == 0:
            nombres = [k for k in recs if not k.startswith("SBP")]
            figura_grilla([orig] + [recs[k] for k in nombres], ["original"] + nombres,
                          DIR_FIG / "reconstruccion_completo.png", columnas=5,
                          suptitulo="Reconstrucción simulada (corte completo)")
            figura_grilla([recorte_nodulo(orig, d)] + [recorte_nodulo(recs[k], d) for k in nombres],
                          ["original"] + nombres, DIR_FIG / "reconstruccion_zoom.png", columnas=5,
                          suptitulo="Reconstrucción simulada (zoom 64 mm al nódulo)")
            fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
            sbp = recs["SBP 180, sin ruido"]
            axs[0].imshow(orig, vmin=-1000, vmax=400, cmap="gray")
            axs[0].set_title("Original")
            axs[1].imshow(sbp, cmap="gray", vmin=np.percentile(sbp, 1), vmax=np.percentile(sbp, 99))
            axs[1].set_title("Retroproyección simple (SBP): halo")
            axs[2].imshow(recs["FBP rampa 180, sin ruido"], vmin=-1000, vmax=400, cmap="gray")
            axs[2].set_title("Retroproyección filtrada (FBP, rampa)")
            for a in axs:
                a.axis("off")
            fig.tight_layout()
            fig.savefig(DIR_FIG / "sbp_vs_fbp.png", dpi=150)
            plt.close(fig)

    df = pd.DataFrame(filas)
    df.to_csv(DIR_RES / "reconstruccion.csv", index=False)
    print(f"\n== Reconstrucción (promedio de {len(archivos)} casos) ==")
    print(df.groupby("condicion", sort=False)[["rmse", "psnr", "ssim"]].mean().round(3).to_string())
    print(f"\nReconstrucciones guardadas en {DIR_CACHE} (las usan filtros.py y robustez.py)")


if __name__ == "__main__":
    main()
