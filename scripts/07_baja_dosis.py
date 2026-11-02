"""Etapa 7 - ¿Los biomarcadores sobreviven a la BAJA DOSIS y a la compresión?

Pregunta (la parte innovadora del trabajo):
El tamizaje de cáncer de pulmón se hace con tomografía de BAJA DOSIS (para
irradiar poco a gente sana). Muchos trabajos de radiómica arman sus modelos
con tomografías de dosis normal, como LIDC. Si una medida cambia cuando baja
la dosis, un modelo armado con dosis normal se equivoca en el tamizaje
aunque en el paper haya dado muy bien. Lo mismo pasa si la imagen se
comprime para mandarla por telemedicina.

Cómo se responde, con la materia (reconstrucción, ruido y compresión):
  1. Cada corte real se pasa por una tomografía simulada con física:
     HU -> μ -> Radon (sinograma) -> Beer-Lambert -> ruido de Poisson con I0
     fotones -> FBP. Menos I0 = menos dosis = más ruido (∝ 1/√I0).
     También se comprime con JPEG (calidad 50 y 25).
  2. Sobre CADA versión se corre el pipeline completo, igual que en la
     clínica: pulmón, crecimiento de regiones desde la semilla, validación
     contra los radiólogos y todas las medidas.
  3. Se mide:
     a. segmentación: ¿sigue siendo aceptable con menos dosis?
     b. reproducibilidad de cada medida: CCC de Lin entre el valor en la
        imagen original y en la degradada (1 = idéntico; a diferencia de la
        correlación, penaliza también que todo se corra para arriba o abajo);
     c. clasificación, de dos maneras:
        - re-armado: el puntaje se arma y evalúa con imágenes de la misma
          dosis (lo mejor que se puede hacer con esa dosis);
        - TRANSFERIDO: el puntaje se arma con las imágenes originales y se
          aplica a las degradadas (lo que pasa en la vida real cuando un
          modelo publicado se usa en un tamizaje). Validación cruzada por
          paciente en los dos casos.

Uso:
  python scripts/07_baja_dosis.py                  # simula (lento) y analiza
  python scripts/07_baja_dosis.py --solo-analisis  # reusa resultados/07_baja_dosis.csv
  opciones: --procesos N, --limite N (para probar), --todos, --sin-calcio

Salidas:
  resultados/07_baja_dosis.csv          medidas de cada nódulo en cada condición
  resultados/07_segmentacion.csv        ruido, Dice y % aceptables por condición
  resultados/07_reproducibilidad.csv    CCC de cada medida en cada condición
  resultados/07_clasificacion.csv       AUC re-armado y transferido por condición
  figuras/07_baja_dosis.png
"""
import argparse
import os
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import compresion as Z  # noqa: E402
from tpi import config as C  # noqa: E402
from tpi.caracteristicas import caracteristicas_interior  # noqa: E402
from tpi.datos import _recortar, _remuestrear, cargar_corte  # noqa: E402
from tpi.entorno import caracteristicas_entorno, mascara_pulmon  # noqa: E402
from tpi.estadistica import META  # noqa: E402
from tpi.puntaje import Puntaje, auc, ic_bootstrap  # noqa: E402
from tpi.reconstruccion import simular  # noqa: E402
from tpi.segmentacion import segmentar_nodulo  # noqa: E402
from tpi.validacion import evaluar  # noqa: E402
from tpi.visualizacion import estimar_ruido_hu  # noqa: E402

PATRON = re.compile(r"(?P<paciente>.+)_s(?P<scan_id>\d+)_n(?P<nodulo>\d+)$")

# (nombre, tipo, parámetro). I0 = fotones por rayo. "dosis ×1" (I0 = 1e5) es
# la referencia de la simulación; ×1/4, ×1/10 y ×1/20 son fracciones de esa
# dosis. La simulación SUMA ruido al que ya trae la imagen real, así que la
# escala comparable entre condiciones es el ruido medido en el parénquima
# (columna ruido_hu), no el nombre.
CONDICIONES = [
    ("original", "original", None),
    ("FBP sin ruido", "fbp", None),
    ("dosis ×1", "fbp", 1e5),
    ("dosis ×1/4", "fbp", 2.5e4),
    ("dosis ×1/10", "fbp", 1e4),
    ("dosis ×1/20", "fbp", 5e3),
    ("JPEG q50", "jpeg", 50),
    ("JPEG q25", "jpeg", 25),
]
ORDEN = [c[0] for c in CONDICIONES]
TAMANO = {"area_mm2", "diametro_eq_mm"}
ENTORNO_EXTRA = {"frac_contacto_no_pulmon", "esqueleto_mm_por_mm_borde", "n_estructuras_convergentes",
                 "convergentes_por_10mm_borde"}


def degradar(d, tipo, param, semilla):
    """Devuelve el corte completo (resolución original, HU) en esa condición."""
    orig = np.maximum(d["corte_completo"], -1000).astype(np.float32)
    if tipo == "original":
        return orig
    if tipo == "fbp":
        rng = np.random.default_rng(semilla)
        return simular(orig, float(d["pixel_mm_original"]), I0=param, rng=rng)[0]
    if tipo == "jpeg":
        img8 = Z.hu_a_8bits(orig)
        return Z.de_8bits_a_hu(Z.comprimir(img8, "JPEG", param)[0])
    raise ValueError(tipo)


def como_si_fuera_original(d, img_completa):
    """Mismo remuestreo y recorte que datos.preparar_corte, con otra imagen."""
    rem = _remuestrear(np.maximum(img_completa, -1024), float(d["pixel_mm_original"]), 1)
    d2 = {k: v for k, v in d.items() if not k.startswith("_")}  # sin cachés del pulmón
    d2["corte_completo_rem"] = rem
    d2["img"] = _recortar(rem, tuple(d["semilla_completo"]), d["img"].shape[0], -1000.0)
    return d2


def procesar(args):
    """Un nódulo en todas las condiciones (corre en un proceso aparte)."""
    ruta, T, R = args
    d = cargar_corte(Path(ruta))
    m = PATRON.match(Path(ruta).stem)
    clave = dict(paciente=m["paciente"], scan_id=int(m["scan_id"]), nodulo=int(m["nodulo"]))
    semilla = zlib.crc32(Path(ruta).stem.encode())  # mismo ruido cada vez que se corre
    filas = []
    for nombre, tipo, param in CONDICIONES:
        img = degradar(d, tipo, param, semilla)
        d2 = como_si_fuera_original(d, img)
        seg, cruda = segmentar_nodulo(d2, umbral=T, devolver_cruda=True, radio_cierre_mm=R)
        fila = dict(**clave, condicion=nombre, ruido_hu=estimar_ruido_hu(d2["img"]),
                    **evaluar(seg, d["consenso"], d["radiologos"]))
        if seg.sum() >= 5:
            fila.update(caracteristicas_interior(d2["img"], seg, mascara_cruda=cruda))
            pulmon = _recortar(mascara_pulmon(d2["corte_completo_rem"]), tuple(d["semilla_completo"]),
                               d["img"].shape[0], False)
            fila.update(caracteristicas_entorno(d2["img"], seg, pulmon)[0])
        filas.append(fila)
    return filas


def parametros():
    ajuste = C.DIR_RESULTADOS / "04_ajuste_umbral.csv"
    if ajuste.exists():
        t = pd.read_csv(ajuste)
        mejor = t.loc[t.dice_medio.idxmax()]
        return int(mejor["umbral_hu"]), float(mejor.get("radio_cierre_mm", C.RADIO_CIERRE_PULMON_MM))
    return C.UMBRAL_HU, C.RADIO_CIERRE_PULMON_MM


def simular_todo(args):
    T, R = parametros()
    rutas = sorted(Path(args.carpeta).glob("*.npz"))
    if args.limite:
        rutas = rutas[:args.limite]
    if not rutas:
        sys.exit("No hay cortes en " + args.carpeta)
    print(f"Simulando {len(rutas)} nódulos × {len(CONDICIONES)} condiciones "
          f"(T = {T} HU, cierre {R:g} mm) con {args.procesos} procesos...")
    filas = []
    with ProcessPoolExecutor(args.procesos) as ex:
        for i, f in enumerate(ex.map(procesar, [(str(r), T, R) for r in rutas]), 1):
            filas += f
            if i % 20 == 0 or i == len(rutas):
                print(f"  {i}/{len(rutas)}", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(C.DIR_RESULTADOS / "07_baja_dosis.csv", index=False)
    return df


def ccc(x, y):
    """Coeficiente de concordancia de Lin."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        return np.nan
    vx, vy = x.var(), y.var()
    den = vx + vy + (x.mean() - y.mean()) ** 2
    return float(2 * np.cov(x, y, ddof=0)[0, 1] / den) if den > 0 else np.nan


def pliegues(pacientes, n=5, semilla=0):
    unicos = np.unique(pacientes)
    np.random.default_rng(semilla).shuffle(unicos)
    asig = {p: i % n for i, p in enumerate(unicos)}
    return np.array([asig[p] for p in pacientes])


def cv_puntaje(X_arma, X_aplica, y, pacientes, k):
    """Validación cruzada por paciente: arma con X_arma (pliegues de
    entrenamiento) y puntúa X_aplica (pliegue de prueba)."""
    pl = pliegues(pacientes)
    s = np.full(len(y), np.nan)
    for f in np.unique(pl):
        tr, te = pl != f, pl == f
        modelo = Puntaje(k).ajustar(X_arma[tr], y[tr])
        s[te] = modelo.puntuar(X_aplica[te])
    return s


def analizar(df, args):
    car = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    pob = car[car.grupo.isin(["benigno", "sospechoso"]) & (car.conjunto == "entrenamiento")]
    if not args.todos:
        pob = pob[pob.aceptable]  # aceptables en la imagen ORIGINAL: misma población en todas las condiciones
    if args.sin_calcio:
        pob = pob[pob.frac_calcio <= C.FRAC_CALCIO_MAX]
    claves = ["paciente", "scan_id", "nodulo"]
    pob = pob[claves + ["grupo", "concordancia_radiologos"]]
    df = df.drop(columns=["concordancia_radiologos"], errors="ignore").merge(pob, on=claves)
    sufijo = "_sincalcio" if args.sin_calcio else ""
    n = df.drop_duplicates(claves).grupo.value_counts().to_dict()
    print(f"\nPoblación: {sum(n.values())} nódulos de entrenamiento {n}"
          f"{' sin calcio' if args.sin_calcio else ''}")

    # a. segmentación
    seg = (df.groupby("condicion").agg(ruido_hu=("ruido_hu", "median"), dice_medio=("dice", "mean"),
                                        aceptables=("aceptable", "mean")).reindex(ORDEN).dropna(how="all"))
    seg.to_csv(C.DIR_RESULTADOS / f"07_segmentacion{sufijo}.csv")
    print("\n== a. Segmentación en cada condición ==")
    print(seg.round(3).to_string())

    # b. reproducibilidad
    medidas = [c for c in df.select_dtypes("number").columns
               if c not in META and c not in {"ruido_hu", "scan_id", "nodulo"}]
    base = df[df.condicion == "original"].set_index(claves)
    rep = {}
    for cond in seg.index:
        if cond == "original":
            continue
        otra = df[df.condicion == cond].set_index(claves).reindex(base.index)
        rep[cond] = {c: ccc(base[c].to_numpy(float), otra[c].to_numpy(float)) for c in medidas}
    rep = pd.DataFrame(rep)
    rep.index.name = "medida"
    rep.to_csv(C.DIR_RESULTADOS / f"07_reproducibilidad{sufijo}.csv")
    col = "dosis ×1/10" if "dosis ×1/10" in rep else rep.columns[-1]
    print(f"\n== b. Reproducibilidad (CCC de Lin vs original; ≥ 0,9 robusta, < 0,7 frágil) ==")
    print(rep.sort_values(col, ascending=False).round(2).to_string())

    # c. clasificación
    entorno = [c for c in medidas if c.startswith("anillo_") or c in ENTORNO_EXTRA]
    interior = [c for c in medidas if c not in entorno]
    conjuntos = {"diámetro solo": (["diametro_eq_mm"], 1),
                 "interior sin tamaño": ([c for c in interior if c not in TAMANO], args.k),
                 "interior": (interior, args.k),
                 "interior + entorno": (interior + entorno, args.k)}
    base = base.reset_index()
    y = (base.grupo == "sospechoso").to_numpy(int)
    filas = []
    for cond in seg.index:
        otra = df[df.condicion == cond].set_index(claves).reindex(base.set_index(claves).index).reset_index()
        for nombre, (cols, k) in conjuntos.items():
            s_re = cv_puntaje(otra[cols], otra[cols], y, base.paciente.to_numpy(), k)
            s_tr = cv_puntaje(base[cols], otra[cols], y, base.paciente.to_numpy(), k)
            lo, hi = ic_bootstrap(y, s_tr, base.paciente)
            filas.append(dict(condicion=cond, puntaje=nombre, auc_rearmado=auc(y, s_re),
                              auc_transferido=auc(y, s_tr), transf_ic95_inf=lo, transf_ic95_sup=hi))
    cla = pd.DataFrame(filas)
    cla.to_csv(C.DIR_RESULTADOS / f"07_clasificacion{sufijo}.csv", index=False)
    print("\n== c. AUC (validación cruzada por paciente) ==")
    print("   re-armado = armado y evaluado con esa dosis; transferido = armado con la original")
    print(cla.pivot(index="condicion", columns="puntaje", values="auc_transferido").reindex(seg.index)
          .round(3).to_string(header=True).replace("puntaje", "TRANSFERIDO"))
    print(cla.pivot(index="condicion", columns="puntaje", values="auc_rearmado").reindex(seg.index)
          .round(3).to_string().replace("puntaje", "RE-ARMADO"))

    figura(seg, rep, cla, C.DIR_FIGURAS / f"07_baja_dosis{sufijo}.png")


def figura(seg, rep, cla, ruta):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
    x = np.arange(len(seg))
    ax[0].plot(x, seg.dice_medio, "o-", label="Dice medio")
    ax[0].plot(x, seg.aceptables, "s--", label="fracción aceptable")
    a2 = ax[0].twinx()
    a2.bar(x, seg.ruido_hu, alpha=0.2, color="gray")
    a2.set_ylabel("ruido en parénquima (HU)", color="gray")
    ax[0].set_xticks(x, seg.index, rotation=40, ha="right", fontsize=7)
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_title("a. Segmentación")

    conds = [c for c in seg.index if c in rep.columns]
    top = rep[conds].sort_values(conds[-1] if "dosis ×1/10" not in conds else "dosis ×1/10", ascending=False)
    top = pd.concat([top.head(12), top.tail(12)]).drop_duplicates()
    im = ax[1].imshow(top.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax[1].set_yticks(range(len(top)), top.index, fontsize=6)
    ax[1].set_xticks(range(len(conds)), conds, rotation=40, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax[1], label="CCC vs original")
    ax[1].set_title("b. Reproducibilidad (12 más robustas / 12 más frágiles)")

    for (nombre, sub), color in zip(cla.groupby("puntaje", sort=False), plt.cm.tab10.colors):
        sub = sub.set_index("condicion").reindex(seg.index)
        ax[2].plot(x, sub.auc_transferido, "o-", color=color, label=f"{nombre} (transferido)")
        ax[2].plot(x, sub.auc_rearmado, ":", color=color, alpha=0.7)
    ax[2].axhline(0.5, color="k", lw=0.8)
    ax[2].set_xticks(x, seg.index, rotation=40, ha="right", fontsize=7)
    ax[2].set_ylabel("AUC")
    ax[2].set_title("c. Clasificación (línea punteada = re-armado)")
    ax[2].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("->", ruta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limite", type=int, default=0, help="solo los primeros N nódulos (para probar)")
    ap.add_argument("--solo-analisis", action="store_true")
    ap.add_argument("--todos", action="store_true", help="incluir segmentaciones no aceptables")
    ap.add_argument("--sin-calcio", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    C.DIR_FIGURAS.mkdir(exist_ok=True)
    ruta = C.DIR_RESULTADOS / "07_baja_dosis.csv"
    df = pd.read_csv(ruta) if args.solo_analisis else simular_todo(args)
    analizar(df, args)


if __name__ == "__main__":
    main()
