"""Etapa 5 - Comparación entre grupos, medida por medida.

Por defecto usa solo ENTRENAMIENTO y solo segmentaciones aceptables.

--rango LO HI: compara solo nódulos de LO a HI mm (diámetro medido por los
radiólogos, que no depende de nuestra segmentación). Es el control por tamaño
más directo: si con nódulos de tamaño parecido una medida igual separa los
grupos, esa medida aporta algo más que el tamaño.

Uso:  python scripts/05_estadistica.py [--todos] [--rango 8 16] [--sin-calcio]
      (--todos: incluye segmentaciones no aceptables)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.estadistica import comparar_grupos, por_estratos  # noqa: E402
from tpi.visualizacion import boxplots  # noqa: E402

COLS = ["medida", "n_sosp", "n_benig", "mediana_sosp", "mediana_benig", "auc", "mayor_en",
        "p_holm", "spearman_parcial_vs_tamano"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--rango", nargs=2, type=float, metavar=("LO", "HI"))
    ap.add_argument("--sin-calcio", action="store_true",
                    help="solo nódulos sin calcificación (frac_calcio <= C.FRAC_CALCIO_MAX)")
    ap.add_argument("--conjunto", default="entrenamiento",
                    help="'entrenamiento' mientras se desarrolla; 'evaluacion' SOLO al final")
    args = ap.parse_args()
    df = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    if args.conjunto != "todos" and df["conjunto"].notna().any():
        df = df[df["conjunto"] == args.conjunto]
    if not args.todos:
        df = df[df["aceptable"]]
    sufijo = "_todos" if args.todos else ""
    if args.sin_calcio:
        df = df[df["frac_calcio"] <= C.FRAC_CALCIO_MAX]
        sufijo += "_sincalcio"
        print(f"Solo nódulos SIN calcio (fracción > {C.HU_CALCIO} HU <= {C.FRAC_CALCIO_MAX:.0%})")
    if args.rango:
        lo, hi = args.rango
        df = df[(df.diametro_medio_mm >= lo) & (df.diametro_medio_mm < hi)]
        sufijo += f"_{lo:g}a{hi:g}mm"
        print(f"Solo nódulos de {lo:g} a {hi:g} mm (diámetro de los radiólogos)")
    if len(df) < 6 or df["grupo"].nunique() < 2:
        sys.exit(f"Muy pocos nódulos para comparar ({len(df)}): revisá el filtro.")
    print(f"n = {len(df)}  ({df.grupo.value_counts().to_dict()})")
    res = comparar_grupos(df)
    res.to_csv(C.DIR_RESULTADOS / f"05_comparacion{sufijo}.csv", index=False)
    print(res[[c for c in COLS if c in res]].head(20).round(3).to_string(index=False))
    sig = res[res["p_holm"] < 0.05]["medida"].tolist() if len(res) else []
    print(f"\nSignificativas después de Holm (p < 0,05): {len(sig)} -> {', '.join(sig) if sig else 'ninguna'}")
    top = res["medida"].head(6).tolist()
    if top:
        boxplots(df, top, C.DIR_FIGURAS / f"05_boxplots{sufijo}.png")
    if not args.rango:
        est = por_estratos(df, None)
        est.to_csv(C.DIR_RESULTADOS / f"05_estratos{sufijo}.csv", index=False)
        if len(est):
            print("\n== Dentro de cada rango de tamaño (las 5 de mayor AUC) ==")
            for e, sub in est.groupby("estrato_mm", sort=False):
                print(f"{e}: n = {int(sub.n_sosp.iloc[0])} sosp / {int(sub.n_benig.iloc[0])} benig")
                print(sub[["medida", "auc", "mayor_en", "p_holm"]].head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
