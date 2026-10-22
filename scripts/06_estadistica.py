"""Etapa 6 - Comparación entre grupos (resultados preliminares).

Uso:  python scripts/06_estadistica.py [--todos]   (--todos: incluye no aceptables)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.estadistica import comparar_grupos, por_estratos  # noqa: E402
from tpi.visualizacion import boxplots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--conjunto", default="entrenamiento",
                    help="'entrenamiento' mientras se desarrolla; 'evaluacion' SOLO al final")
    args = ap.parse_args()
    df = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    if args.conjunto != "todos" and df["conjunto"].notna().any():
        df = df[df["conjunto"] == args.conjunto]
    if not args.todos:
        df = df[df["aceptable"]]
    print(f"n = {len(df)}  ({df.grupo.value_counts().to_dict()})")
    res = comparar_grupos(df)
    sufijo = "_todos" if args.todos else ""
    res.to_csv(C.DIR_RESULTADOS / f"06_comparacion{sufijo}.csv", index=False)
    print(res.head(15).round(4).to_string())
    top = res["medida"].head(6).tolist()
    if top:
        boxplots(df, top, C.DIR_FIGURAS / f"06_boxplots{sufijo}.png")
        est = por_estratos(df, top)
        est.to_csv(C.DIR_RESULTADOS / f"06_estratos{sufijo}.csv", index=False)


if __name__ == "__main__":
    main()
