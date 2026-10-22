"""Etapa 1 - Contextualización y selección de nódulos.

No necesita imágenes: usa solo la base de anotaciones que trae pylidc.

Salidas:
  resultados/candidatos_todos.csv      todos los nódulos con motivo de exclusión
  resultados/nodulos_seleccionados.csv los incluidos, con conjunto train/eval
  resultados/flujo_seleccion.csv       cuántos se pierden en cada filtro
  figuras/01_contextualizacion.png

Uso:  python scripts/01_seleccionar_nodulos.py [--max-pacientes 80]
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
from tpi.datos import dividir_por_paciente, tabla_candidatos  # noqa: E402


def submuestrear(df, max_pacientes, semilla=C.SEMILLA):
    """Elige pacientes priorizando que ambos grupos queden balanceados."""
    rng = np.random.default_rng(semilla)
    por_pac = df.groupby("paciente")["grupo"].agg(lambda g: (g == "sospechoso").mean())
    sosp = por_pac[por_pac >= 0.5].index.to_numpy()
    benig = por_pac[por_pac < 0.5].index.to_numpy()
    rng.shuffle(sosp)
    rng.shuffle(benig)
    mitad = max_pacientes // 2
    elegidos = list(sosp[:mitad]) + list(benig[:max_pacientes - min(mitad, len(sosp))])
    return df[df["paciente"].isin(elegidos[:max_pacientes])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pacientes", type=int, default=0,
                    help="0 = todos los que cumplen criterios")
    ap.add_argument("--rehacer", action="store_true", help="recalcular candidatos_todos.csv")
    args = ap.parse_args()
    C.DIR_RESULTADOS.mkdir(exist_ok=True)
    C.DIR_FIGURAS.mkdir(exist_ok=True)

    cache = C.DIR_RESULTADOS / "candidatos_todos.csv"
    if cache.exists() and not args.rehacer:
        todos = pd.read_csv(cache, keep_default_na=False, na_values=[""])
        todos["motivo_exclusion"] = todos["motivo_exclusion"].fillna("")
    else:  # recorre las 1018 tomografías (~4 min)
        todos = tabla_candidatos()
        todos.to_csv(cache, index=False)

    # Flujo de selección (para el informe): se aplican los filtros en orden.
    orden = ["espesor > 2.5 mm", "agrupación ambigua (> 4 anotaciones)", "< 3 radiólogos", "no sólido", "< 6 mm", "malignidad intermedia"]
    restantes = len(todos)
    flujo = [dict(paso="nódulos agrupados (todas las tomografías)", quedan=restantes)]
    for motivo in orden:
        restantes -= int((todos["motivo_exclusion"] == motivo).sum())
        flujo.append(dict(paso=f"excluir: {motivo}", quedan=restantes))
    flujo = pd.DataFrame(flujo)
    flujo.to_csv(C.DIR_RESULTADOS / "flujo_seleccion.csv", index=False)
    print(flujo.to_string(index=False))

    sel = todos[todos["incluido"]].drop(columns=["incluido", "motivo_exclusion"])
    if args.max_pacientes:
        print(f"\nCumplen criterios: {len(sel)} nódulos de {sel.paciente.nunique()} pacientes "
              f"-> submuestra de {args.max_pacientes} pacientes")
        sel = submuestrear(sel, args.max_pacientes)
    sel = dividir_por_paciente(sel)
    sel.to_csv(C.DIR_RESULTADOS / "nodulos_seleccionados.csv", index=False)

    print(f"\nSeleccionados: {len(sel)} nódulos de {sel.paciente.nunique()} pacientes")
    print(pd.crosstab(sel["grupo"], sel["conjunto"], margins=True))
    print("\nMalignidad mediana por grupo:\n", pd.crosstab(sel["grupo"], sel["malignidad_mediana"]))

    # --- Figura de contextualización ---
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.3))
    por_scan = todos.drop_duplicates("scan_id")
    axs[0].hist(por_scan["pixel_mm"], bins=25, color="gray")
    axs[0].set_title("Tamaño de píxel (todas)")
    axs[0].set_xlabel("mm")
    axs[1].hist(por_scan["espesor_mm"], bins=np.arange(0.5, 5.5, 0.25), color="gray")
    axs[1].axvline(C.MAX_ESPESOR_MM, color="r", ls="--")
    axs[1].set_title("Espesor de corte (todas)")
    axs[1].set_xlabel("mm")
    for g, col in (("benigno", "tab:blue"), ("sospechoso", "tab:red")):
        axs[2].hist(sel.loc[sel.grupo == g, "diametro_medio_mm"], bins=15, alpha=0.6, color=col, label=g)
    axs[2].set_title("Diámetro (seleccionados)")
    axs[2].set_xlabel("mm")
    axs[2].legend()
    m = todos[todos["n_anotaciones"] >= 3]
    axs[3].hist(m["malignidad_mediana"], bins=np.arange(0.75, 5.5, 0.5), color="gray")
    axs[3].axvspan(2.25, 3.75, color="k", alpha=0.1)
    axs[3].set_title("Malignidad mediana (≥3 radiólogos)")
    fig.tight_layout()
    fig.savefig(C.DIR_FIGURAS / "01_contextualizacion.png", dpi=150)
    print("\nOK ->", C.DIR_RESULTADOS)


if __name__ == "__main__":
    main()
