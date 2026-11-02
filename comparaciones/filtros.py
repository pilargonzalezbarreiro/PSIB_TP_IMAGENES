"""¿Qué filtro conviene para limpiar una tomografía de baja dosis?

Se filtra la reconstrucción con 1/10 de la dosis (FBP rampa, la más ruidosa)
y se compara contra la imagen original con cuatro números:

  RMSE (HU) ............ error promedio contra la original (menos = mejor)
  SSIM ................. parecido estructural, 0 a 1 (más = mejor)
  ruido residual (HU) .. cuánto ruido queda en el parénquima, zona pareja
                         donde todo lo que varía es ruido (menos = mejor)
  nitidez del borde (%)  gradiente medio (Sobel) sobre el borde del nódulo,
                         relativo a la imagen original: 100 % = bordes
                         intactos, menos = bordes borroneados

Los dos últimos separan lo que RMSE mezcla: un filtro puede bajar el ruido
borroneando bordes (gaussiano fuerte) o conservar bordes dejando ruido
(Wiener, mediana). El gráfico ruido-vs-nitidez muestra ese compromiso; cada
filtro se prueba con varios tamaños porque son cuentas baratas (milisegundos).

Usa la reconstrucción guardada por reconstruccion.py; si no está, calcula
solo la que necesita.

Salidas:
  resultados/comparaciones/filtros.csv
  figuras/comparaciones/filtros_zoom.png, filtros_compromiso.png

Uso:  python comparaciones/filtros.py [--casos 3] [--carpeta datos/cortes]
"""
from _comun import (BAJA, DIR_FIG, DIR_RES, RANGO_HU, C, argumentos, cargar_cache, cargar_corte,
                    listar_casos, mascara_nodulo, np, original, pd, recorte_nodulo, roi_trabajo)

import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from tpi import compresion as Z  # noqa: E402
from tpi import mejoramiento as M  # noqa: E402
from tpi import reconstruccion as R  # noqa: E402
from tpi.visualizacion import figura_grilla  # noqa: E402

CLAVE = f"FBP rampa 180, I0={BAJA:g}"

# (nombre, función). Los tamaños del medio son los "por defecto" del pipeline.
VARIANTES = [
    ("mediana 3x3", lambda x: M.mediana(x, 3)),
    ("mediana 5x5", lambda x: M.mediana(x, 5)),
    ("gaussiano σ=0.5", lambda x: M.gaussiano(x, 0.5)),
    ("gaussiano σ=1", lambda x: M.gaussiano(x, 1.0)),
    ("gaussiano σ=2", lambda x: M.gaussiano(x, 2.0)),
    ("Wiener 3x3", lambda x: M.filtro_wiener(x, 3)),
    ("Wiener 5x5", lambda x: M.filtro_wiener(x, 5)),
    ("Wiener 7x7", lambda x: M.filtro_wiener(x, 7)),
]


def ruido_parenquima(img_roi, ref_roi):
    """Desvío (robusto) de la diferencia con la original en el parénquima."""
    zona = (ref_roi > -950) & (ref_roi < -700)
    r = (img_roi - ref_roi)[zona]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def nitidez_borde(img_roi, mascara):
    borde = mascara & ~ndi.binary_erosion(mascara)
    borde = ndi.binary_dilation(borde)  # banda de ~2 px sobre el contorno
    g = np.hypot(ndi.sobel(img_roi, 0), ndi.sobel(img_roi, 1))
    return float(g[borde].mean())


def main():
    args = argumentos(__doc__.splitlines()[0]).parse_args()
    archivos = listar_casos(args.carpeta, args.casos)
    filas = []
    for n, ruta in enumerate(archivos):
        d = cargar_corte(ruta)
        orig = original(d)
        cache = cargar_cache(ruta)
        if cache is not None and CLAVE in cache:
            ruidosa = cache[CLAVE]
        else:
            print(f"[{ruta.stem}] sin reconstrucción guardada: calculando solo la de baja dosis")
            ruidosa, _ = R.simular(orig, float(d["pixel_mm_original"]), I0=BAJA,
                                   rng=np.random.default_rng(C.SEMILLA))
        fov = R.mascara_fov(orig.shape[0])
        mascara = mascara_nodulo(d)
        ref_roi = roi_trabajo(orig, d)
        nit_ref = nitidez_borde(ref_roi, mascara)

        versiones = {"sin filtro": ruidosa, **{k: f(ruidosa) for k, f in VARIANTES}}
        for nombre, img in versiones.items():
            roi = roi_trabajo(img, d)
            m = Z.metricas(np.clip(orig, -1000, 400), np.clip(img, -1000, 400), RANGO_HU, fov)
            filas.append(dict(caso=ruta.stem, filtro=nombre, **m,
                              ruido_residual_hu=ruido_parenquima(roi, ref_roi),
                              nitidez_borde_pct=100 * nitidez_borde(roi, mascara) / nit_ref))

        if n == 0:
            elegidos = ["sin filtro", "mediana 3x3", "gaussiano σ=1", "Wiener 5x5"]
            figura_grilla([recorte_nodulo(orig, d)] + [recorte_nodulo(versiones[k], d) for k in elegidos],
                          ["original"] + elegidos, DIR_FIG / "filtros_zoom.png", columnas=5,
                          suptitulo=f"Filtros sobre FBP de baja dosis (I0={BAJA:g})")

    df = pd.DataFrame(filas)
    df.to_csv(DIR_RES / "filtros.csv", index=False)
    res = df.groupby("filtro", sort=False)[["rmse", "psnr", "ssim", "ruido_residual_hu", "nitidez_borde_pct"]].mean()
    print(f"== Filtros sobre baja dosis (promedio de {len(archivos)} casos) ==")
    print(res.round(2).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colores = {"mediana": "tab:green", "gaussiano": "tab:blue", "Wiener": "tab:orange", "sin": "k"}
    for familia, col in colores.items():
        sub = res[[k.startswith(familia) for k in res.index]]
        ax.plot(sub["ruido_residual_hu"], sub["nitidez_borde_pct"], "o-", color=col,
                label=familia if familia != "sin" else "sin filtro")
        for k, r in sub.iterrows():
            ax.annotate(k if familia == "sin" else k.split(" ", 1)[-1],
                        (r["ruido_residual_hu"], r["nitidez_borde_pct"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_xlabel("ruido residual en el parénquima (HU)  ← mejor")
    ax.set_ylabel("nitidez del borde del nódulo (% de la original)  ↑ mejor")
    ax.set_title("Compromiso ruido vs bordes (arriba a la izquierda = ideal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_FIG / "filtros_compromiso.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
