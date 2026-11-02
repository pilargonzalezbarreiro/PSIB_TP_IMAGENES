"""Etapa 6 - Puntaje combinado y curvas ROC (objetivo intermedio del anteproyecto).

Compara cuatro puntajes, cada uno armado con un conjunto de medidas:
  diámetro solo .......... la referencia: lo que ya usa Lung-RADS
  interior sin tamaño .... forma, borde, intensidad, composición y textura,
                           sin área ni diámetro: ¿hay algo más allá del tamaño?
  interior ............... todas las medidas de adentro del nódulo (hipótesis 1)
  interior + entorno ..... más los anillos y las estructuras convergentes
                           (hipótesis 2: ¿el entorno agrega?)
  entorno solo ........... solo anillos y convergencia: ¿tiene señal propia?

Por defecto solo usa ENTRENAMIENTO, con validación cruzada por paciente.
El 20 % de evaluación se usa una sola vez, al final, con --evaluar: si se
mira antes y se cambia algo según lo que dio, deja de ser una evaluación.

Salidas:
  resultados/06_puntaje.csv            AUC de cada puntaje con IC 95 %
  resultados/06_medidas_elegidas.csv   qué medidas eligió cada puntaje y con qué signo
  figuras/06_roc.png

--rango LO HI: arma y evalúa los puntajes solo con nódulos de LO a HI mm
(diámetro de los radiólogos). Ahí el diámetro casi no separa, así que se ve
si el resto de las medidas aporta algo más allá del tamaño.

--sin-calcio: solo nódulos sin calcificación. El calcio es un signo de
benignidad que los radiólogos ya usan (y Lung-RADS ya resuelve); la pregunta
clínica abierta son los nódulos sólidos sin calcio.

Uso:  python scripts/06_puntaje.py [--k 5] [--todos] [--rango 8 16] [--sin-calcio] [--evaluar]
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

from tpi import config as C  # noqa: E402
from tpi.estadistica import META  # noqa: E402
from tpi.puntaje import Puntaje, auc, curva_roc, ic_bootstrap, validacion_cruzada  # noqa: E402

TAMANO = {"area_mm2", "diametro_eq_mm"}
ENTORNO_EXTRA = {"frac_contacto_no_pulmon", "esqueleto_mm_por_mm_borde", "n_estructuras_convergentes",
                 "convergentes_por_10mm_borde"}


def conjuntos(df):
    medidas = [c for c in df.select_dtypes("number").columns if c not in META]
    entorno = [c for c in medidas if c.startswith("anillo_") or c in ENTORNO_EXTRA]
    interior = [c for c in medidas if c not in entorno]
    return {
        "diámetro solo": ["diametro_eq_mm"],
        "interior sin tamaño": [c for c in interior if c not in TAMANO],
        "interior": interior,
        "interior + entorno": interior + entorno,
        # El top-k elige las k mejores medidas: si las del entorno son más
        # débiles que las del interior, "interior + entorno" termina igual a
        # "interior". "entorno solo" muestra si el entorno tiene señal propia.
        "entorno solo": entorno,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="cuántas medidas combina cada puntaje")
    ap.add_argument("--todos", action="store_true", help="incluir segmentaciones no aceptables")
    ap.add_argument("--evaluar", action="store_true", help="evaluar en el 20 %% (usar UNA vez, al final)")
    ap.add_argument("--rango", nargs=2, type=float, metavar=("LO", "HI"))
    ap.add_argument("--sin-calcio", action="store_true",
                    help="solo nódulos sin calcificación (frac_calcio <= C.FRAC_CALCIO_MAX)")
    args = ap.parse_args()

    df = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    df = df[df["grupo"].isin(["benigno", "sospechoso"])]
    if not args.todos:
        df = df[df["aceptable"]]
    sufijo = ""
    if args.sin_calcio:
        df = df[df["frac_calcio"] <= C.FRAC_CALCIO_MAX]
        sufijo += "_sincalcio"
        print(f"Solo nódulos SIN calcio (fracción > {C.HU_CALCIO} HU <= {C.FRAC_CALCIO_MAX:.0%})")
    if args.rango:
        lo, hi = args.rango
        df = df[(df.diametro_medio_mm >= lo) & (df.diametro_medio_mm < hi)]
        sufijo += f"_{lo:g}a{hi:g}mm"
        print(f"Solo nódulos de {lo:g} a {hi:g} mm (diámetro de los radiólogos)")
    tr = df[df["conjunto"].fillna("entrenamiento") == "entrenamiento"].reset_index(drop=True)
    ev = df[df["conjunto"] == "evaluacion"].reset_index(drop=True)
    if len(tr) < 6 or tr["grupo"].nunique() < 2:
        sys.exit(f"Muy pocos nódulos de entrenamiento ({len(tr)}): revisá el filtro.")
    y_tr = (tr["grupo"] == "sospechoso").to_numpy(int)
    print(f"Entrenamiento: {len(tr)} nódulos de {tr.paciente.nunique()} pacientes "
          f"({int(y_tr.sum())} sospechosos)")
    if len(tr) < 10:
        print("Ojo: con tan pocos nódulos las AUC son muy inestables (mirar los intervalos).")

    filas, elegidas, rocs = [], [], {}
    for nombre, cols in conjuntos(df).items():
        k = 1 if nombre == "diámetro solo" else args.k
        s_cv = validacion_cruzada(tr[cols], y_tr, tr["paciente"], k=k)
        a = auc(y_tr, s_cv)
        lo, hi = ic_bootstrap(y_tr, s_cv, tr["paciente"])
        fila = dict(puntaje=nombre, n_medidas_disponibles=len(cols), k=k,
                    auc_validacion_cruzada=a, ic95_inf=lo, ic95_sup=hi)
        final = Puntaje(k).ajustar(tr[cols], y_tr)
        elegidas.append(final.medidas.assign(puntaje=nombre))
        rocs[nombre] = curva_roc(y_tr, np.nan_to_num(s_cv, nan=0.0))
        if args.evaluar and len(ev):
            y_ev = (ev["grupo"] == "sospechoso").to_numpy(int)
            s_ev = final.puntuar(ev[cols])
            fila["auc_evaluacion"] = auc(y_ev, s_ev)
            fila["ev_ic95_inf"], fila["ev_ic95_sup"] = ic_bootstrap(y_ev, s_ev, ev["paciente"])
        filas.append(fila)

    res = pd.DataFrame(filas)
    res.to_csv(C.DIR_RESULTADOS / f"06_puntaje{sufijo}.csv", index=False)
    med = pd.concat(elegidas, ignore_index=True)[["puntaje", "medida", "auc_entrenamiento", "signo"]]
    med.to_csv(C.DIR_RESULTADOS / f"06_medidas_elegidas{sufijo}.csv", index=False)
    print("\n== AUC de cada puntaje ==")
    print(res.round(3).to_string(index=False))
    print("\n== Medidas que eligió cada puntaje (signo +1 = mayor en sospechosos) ==")
    print(med.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.2, 5))
    for (nombre, (fpr, tpr)), fila in zip(rocs.items(), filas):
        ax.plot(fpr, tpr, lw=1.8, label=f"{nombre} (AUC {fila['auc_validacion_cruzada']:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("1 - especificidad (benignos marcados como sospechosos)")
    ax.set_ylabel("sensibilidad (sospechosos detectados)")
    ax.set_title("Curvas ROC (validación cruzada en entrenamiento)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.DIR_FIGURAS / f"06_roc{sufijo}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
