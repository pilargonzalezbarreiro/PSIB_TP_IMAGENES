"""¿Las medidas del nódulo sobreviven a cómo se adquiere y se guarda la imagen?

La materia avisa que filtrar y comprimir con pérdida cambian la textura.
Acá se cuantifica: se segmenta el nódulo UNA vez en la imagen original y,
con esa misma máscara, se miden intensidad y textura (GLCM) en cada versión
de la imagen (cada reconstrucción, cada filtro, cada JPEG). Si una medida
casi no cambia, es robusta; si cambia mucho, sus valores dependen más de la
adquisición que del nódulo (hipótesis 3).

Intensidad: cambio en HU (la media del nódulo ronda 0 HU, un % no tendría
sentido). Textura: cambio en % (son números sin unidad).

Necesita las reconstrucciones guardadas por reconstruccion.py.

Salidas:
  resultados/comparaciones/robustez.csv
  figuras/comparaciones/robustez.png

Uso:  python comparaciones/robustez.py [--casos 3] [--carpeta datos/cortes]
"""
import sys

from _comun import (BAJA, DIR_FIG, DIR_RES, C, argumentos, cargar_cache, cargar_corte, listar_casos,
                    mascara_nodulo, medidas_nodulo, np, original, pd)

import matplotlib.pyplot as plt  # noqa: E402

from tpi import compresion as Z  # noqa: E402
from tpi import mejoramiento as M  # noqa: E402

MEDIDAS_HU = ["hu_media", "hu_desvio"]
MEDIDAS_GLCM = ["glcm_energia", "glcm_entropia", "glcm_contraste", "glcm_homogeneidad",
                "glcm_correlacion"]


def main():
    args = argumentos(__doc__.splitlines()[0]).parse_args()
    archivos = listar_casos(args.carpeta, args.casos)
    filas = []
    for ruta in archivos:
        d = cargar_corte(ruta)
        cache = cargar_cache(ruta)
        if cache is None:
            sys.exit(f"Falta la reconstrucción de {ruta.stem}: corré antes comparaciones/reconstruccion.py")
        orig = original(d)
        mascara = mascara_nodulo(d)
        ref = medidas_nodulo(orig, d, mascara)

        versiones = [("reconstrucción", k, v) for k, v in cache.items() if not k.startswith("SBP")]
        ruidosa = cache[f"FBP rampa 180, I0={BAJA:g}"]
        versiones += [("filtro sobre baja dosis", k, f(ruidosa)) for k, f in M.FILTROS.items()]
        versiones += [("filtro sobre la original", k, f(orig)) for k, f in M.FILTROS.items()]
        img8 = Z.hu_a_8bits(orig)
        versiones.append(("compresión", "ventana a 8 bits", Z.de_8bits_a_hu(img8)))
        versiones += [("compresión", f"JPEG q{q}", Z.de_8bits_a_hu(Z.comprimir(img8, "JPEG", q)[0]))
                      for q in C.CALIDADES_JPEG]

        for etapa, nombre, img in versiones:
            m = medidas_nodulo(img, d, mascara)
            fila = dict(caso=ruta.stem, etapa=etapa, condicion=nombre)
            for k in MEDIDAS_HU:
                fila[f"d_{k}_HU"] = m[k] - ref[k]
            for k in MEDIDAS_GLCM:
                fila[f"{k}_%"] = 100 * (m[k] - ref[k]) / abs(ref[k]) if ref[k] else np.nan
            filas.append(fila)

    df = pd.DataFrame(filas)
    df.to_csv(DIR_RES / "robustez.csv", index=False)
    res = df.groupby(["etapa", "condicion"], sort=False).mean(numeric_only=True)
    print(f"== Robustez (promedio de {len(archivos)} casos): intensidad en HU, textura en % ==")
    print(res.round(1).to_string())

    cols = [f"{k}_%" for k in MEDIDAS_GLCM]
    tabla = res[cols]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(tabla) + 1.5))
    im = ax.imshow(tabla.abs().to_numpy(), cmap="Reds", vmin=0, vmax=50, aspect="auto")
    ax.set_yticks(range(len(tabla)))
    ax.set_yticklabels([f"{e}: {c}" for e, c in tabla.index], fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(MEDIDAS_GLCM, rotation=35, ha="right", fontsize=7)
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            ax.text(j, i, f"{tabla.iloc[i, j]:+.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="|cambio| % vs original")
    ax.set_title("¿Cuánto cambia la textura del nódulo? (misma máscara, % vs original)")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "robustez.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
