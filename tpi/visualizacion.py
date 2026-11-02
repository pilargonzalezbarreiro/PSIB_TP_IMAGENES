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


def estimar_ruido_hu(img):
    """Ruido ≈ desvío del residuo (imagen - suavizada) en el parénquima
    (zona casi uniforme, HU entre -950 y -700). Se usa la MAD (desvío robusto)
    para que los vasos no inflen la estimación."""
    from scipy import ndimage as ndi
    res = img - ndi.gaussian_filter(img, 2)
    m = (img > -950) & (img < -700)
    if m.sum() < 50:
        return np.nan
    r = res[m]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def figura_analisis(cortes, ruta=None):
    """Paso de ANÁLISIS de la materia: antes de segmentar, mirar los
    histogramas. Cada línea es el histograma de la región de 64 mm de un
    nódulo: si son bimodales (aire del pulmón vs tejido) se justifica separar
    por umbral; el umbral de Otsu de cada uno se marca abajo."""
    from skimage.filters import threshold_otsu
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    bins = np.linspace(-1100, 500, 81)
    otsus, ruidos = [], []
    for d in cortes:
        h, _ = np.histogram(d["img"], bins=bins, density=True)
        axs[0].plot((bins[:-1] + bins[1:]) / 2, h, alpha=0.5, lw=1)
        otsus.append(threshold_otsu(np.clip(d["img"], -1000, 400)))
        ruidos.append(estimar_ruido_hu(d["img"]))
    axs[0].set_yscale("log")
    axs[0].plot(otsus, np.full(len(otsus), axs[0].get_ylim()[0] * 1.5), "k|", ms=12, label="umbral de Otsu")
    for v, t in ((-850, "pulmón"), (0, "tejido")):
        axs[0].axvline(v, color="gray", ls=":")
        axs[0].text(v + 15, axs[0].get_ylim()[1] * 0.3, t, fontsize=8)
    axs[0].set_xlabel("HU")
    axs[0].set_title("Histogramas de las regiones (uno por nódulo): bimodales")
    axs[0].legend(fontsize=8)
    axs[1].hist([r for r in ruidos if np.isfinite(r)], bins=12, color="gray")
    axs[1].set_xlabel("ruido estimado en el parénquima (HU)")
    axs[1].set_title("Ruido de las imágenes originales (granulado)")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
    print(f"Análisis: Otsu medio {np.mean(otsus):.0f} HU (rango {np.min(otsus):.0f} a {np.max(otsus):.0f}); "
          f"ruido mediano en parénquima {np.nanmedian(ruidos):.0f} HU")
    return fig
