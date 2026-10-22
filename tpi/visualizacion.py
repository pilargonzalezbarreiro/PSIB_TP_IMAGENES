"""Figuras del informe."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VENTANA_PULMON = dict(vmin=-1000, vmax=400, cmap="gray")


def _contorno(ax, m, color, lw=1.2, ls="-"):
    if m is not None and m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def figura_caso(img, consenso, radiologos, alg, semilla, titulo="", anillos=None, ruta=None):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    axs[0].imshow(img, **VENTANA_PULMON)
    axs[0].plot(semilla[1], semilla[0], "c+", ms=10)
    axs[0].set_title("ROI + semilla")
    axs[1].imshow(img, **VENTANA_PULMON)
    for m in radiologos:
        _contorno(axs[1], m, "yellow", 0.8, "--")
    _contorno(axs[1], consenso, "lime", 1.5)
    _contorno(axs[1], alg, "red", 1.5)
    axs[1].set_title("radiólogos (--), consenso (verde), algoritmo (rojo)", fontsize=8)
    axs[2].imshow(img, **VENTANA_PULMON)
    _contorno(axs[2], alg, "red", 1.2)
    if anillos is not None:
        rgba = np.zeros(img.shape + (4,))
        rgba[anillos[0]] = (0, 0.6, 1, 0.45)
        rgba[anillos[1]] = (1, 0.6, 0, 0.45)
        axs[2].imshow(rgba)
    axs[2].set_title("anillos 0-3 mm (azul) y 3-6 mm (naranja)", fontsize=8)
    for a in axs:
        a.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
    return fig


def figura_grilla(imagenes, titulos, ruta=None, ventana=VENTANA_PULMON, columnas=4, suptitulo=""):
    filas = int(np.ceil(len(imagenes) / columnas))
    fig, axs = plt.subplots(filas, columnas, figsize=(3.2 * columnas, 3.3 * filas))
    for a in np.ravel(axs):
        a.axis("off")
    for a, im, t in zip(np.ravel(axs), imagenes, titulos):
        a.imshow(im, **ventana)
        a.set_title(t, fontsize=8)
    fig.suptitle(suptitulo, fontsize=10)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
    return fig


def boxplots(df, medidas, ruta=None):
    n = len(medidas)
    fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
    for a, m in zip(np.atleast_1d(axs), medidas):
        datos = [df.loc[df.grupo == g, m].dropna() for g in ("benigno", "sospechoso")]
        a.boxplot(datos, tick_labels=["benigno", "sosp."], widths=0.6)
        a.set_title(m, fontsize=8)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
    return fig
