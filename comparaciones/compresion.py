"""¿Cómo conviene guardar o enviar las tomografías?

Para cada corte: ventana a 8 bits, y después cada método de la materia
(Huffman, RLE, PNG sin pérdida; JPEG con pérdida a varias calidades). Se mide
cuánto achica (tasa = tamaño sin comprimir / tamaño comprimido) y cuánto se
pierde (RMSE en HU y SSIM contra la original).

No depende de la reconstrucción: corre solo y en segundos.

Salidas:
  resultados/comparaciones/compresion.csv
  figuras/comparaciones/compresion_zoom.png, error_jpeg_q25.png

Uso:  python comparaciones/compresion.py [--casos 3] [--carpeta datos/cortes]
"""
from _comun import (DIR_FIG, DIR_RES, RANGO_HU, C, argumentos, cargar_corte, listar_casos, np,
                    original, pd, recorte_nodulo)

import matplotlib.pyplot as plt  # noqa: E402

from tpi import compresion as Z  # noqa: E402
from tpi.visualizacion import figura_grilla  # noqa: E402


def main():
    args = argumentos(__doc__.splitlines()[0], casos_por_defecto=10).parse_args()
    archivos = listar_casos(args.carpeta, args.casos)
    filas = []
    for n, ruta in enumerate(archivos):
        d = cargar_corte(ruta)
        orig = original(d)
        img8 = Z.hu_a_8bits(orig)
        crudo = img8.size  # 1 byte por píxel
        hu8 = Z.de_8bits_a_hu(img8)
        m8 = Z.metricas(np.clip(orig, -1000, 400), hu8, RANGO_HU)
        base = dict(caso=ruta.stem)
        filas.append(dict(base, formato="8 bits sin comprimir (ventana)", con_perdida="solo la ventana",
                          bytes=crudo, tasa=1.0, **m8))
        for nombre, nb in (("Huffman", Z.bytes_huffman(img8)), ("RLE", Z.bytes_rle(img8))):
            filas.append(dict(base, formato=nombre, con_perdida="no", bytes=nb, tasa=crudo / nb, **m8))
        png, nb = Z.comprimir(img8, "PNG")
        assert np.array_equal(png, img8), "PNG debería ser sin pérdida"
        filas.append(dict(base, formato="PNG", con_perdida="no", bytes=nb, tasa=crudo / nb, **m8))
        jpgs = {}
        for q in C.CALIDADES_JPEG:
            jpg, nb = Z.comprimir(img8, "JPEG", q)
            jpgs[q] = Z.de_8bits_a_hu(jpg)
            mj = Z.metricas(np.clip(orig, -1000, 400), jpgs[q], RANGO_HU)
            filas.append(dict(base, formato=f"JPEG q{q}", con_perdida="sí", bytes=nb, tasa=crudo / nb, **mj))

        if n == 0:
            figura_grilla([recorte_nodulo(orig, d)] + [recorte_nodulo(v, d) for v in jpgs.values()],
                          ["original"] + [f"JPEG q{q}" for q in jpgs], DIR_FIG / "compresion_zoom.png",
                          columnas=5, suptitulo="Compresión JPEG (zoom al nódulo)")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(recorte_nodulo(jpgs[25] - hu8, d), cmap="RdBu", vmin=-60, vmax=60)
            ax.set_title("Error JPEG q25 (HU): bloques 8x8")
            ax.axis("off")
            fig.savefig(DIR_FIG / "error_jpeg_q25.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(filas)
    df.to_csv(DIR_RES / "compresion.csv", index=False)
    print(f"== Compresión (promedio de {len(archivos)} casos) ==")
    print(df.groupby(["formato", "con_perdida"], sort=False)[["bytes", "tasa", "rmse", "psnr", "ssim"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
