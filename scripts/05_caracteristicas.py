"""Etapa 5 - Segmentación, validación y extracción de características.

1. (opcional, --ajustar) barre el umbral T en el conjunto de ENTRENAMIENTO
   y elige el que maximiza el Dice medio contra el consenso.
2. Segmenta cada nódulo desde la semilla (y con Otsu, como comparación).
3. Valida contra el consenso con el umbral de aceptabilidad.
4. Extrae medidas del interior y del entorno.

Salidas:
  resultados/caracteristicas.csv   una fila por nódulo
  resultados/05_ajuste_umbral.csv  (si --ajustar)
  figuras/05_caso_*.png            algunos casos con contornos y anillos

Uso:  python scripts/05_caracteristicas.py [--ajustar] [--carpeta datos/cortes]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.caracteristicas import caracteristicas_interior  # noqa: E402
from tpi.datos import _recortar, cargar_corte  # noqa: E402
from tpi.entorno import caracteristicas_entorno, mascara_pulmon  # noqa: E402
from tpi.segmentacion import crecimiento_regiones, segmentar_otsu  # noqa: E402
from tpi.validacion import dice, evaluar  # noqa: E402
from tpi.visualizacion import figura_caso  # noqa: E402

PATRON = re.compile(r"(?P<paciente>.+)_s(?P<scan_id>\d+)_n(?P<nodulo>\d+)$")


def cargar_todo(carpeta, meta):
    casos = []
    for ruta in sorted(Path(carpeta).glob("*.npz")):
        m = PATRON.match(ruta.stem)
        if not m:
            continue
        clave = dict(paciente=m["paciente"], scan_id=int(m["scan_id"]), nodulo=int(m["nodulo"]))
        fila = meta[(meta.scan_id == clave["scan_id"]) & (meta.nodulo == clave["nodulo"])]
        casos.append((ruta.stem, clave, fila.iloc[0].to_dict() if len(fila) else {}, cargar_corte(ruta)))
    return casos


def pulmon_roi(d):
    p = mascara_pulmon(d["corte_completo_rem"])
    lado = d["img"].shape[0]
    return _recortar(p, tuple(d["semilla_completo"]), lado, False)


def ajustar_umbral(casos):
    filas = []
    entren = [c for c in casos if c[2].get("conjunto", "entrenamiento") == "entrenamiento"]
    for T in range(-600, -275, 25):
        ds = [dice(crecimiento_regiones(d["img"], d["semilla"], umbral=T), d["consenso"])
              for *_, d in entren]
        filas.append(dict(umbral_hu=T, dice_medio=np.mean(ds), dice_mediana=np.median(ds), n=len(ds)))
    df = pd.DataFrame(filas)
    df.to_csv(C.DIR_RESULTADOS / "05_ajuste_umbral.csv", index=False)
    print(df.round(3).to_string(index=False))
    return int(df.loc[df.dice_medio.idxmax(), "umbral_hu"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(C.DIR_CORTES))
    ap.add_argument("--meta", default=str(C.DIR_RESULTADOS / "nodulos_seleccionados.csv"))
    ap.add_argument("--ajustar", action="store_true")
    ap.add_argument("--figuras", type=int, default=6)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta) if Path(args.meta).exists() else pd.DataFrame(columns=["scan_id", "nodulo"])
    casos = cargar_todo(args.carpeta, meta)
    if not casos:
        sys.exit("No hay cortes en " + args.carpeta)
    C.DIR_RESULTADOS.mkdir(exist_ok=True)
    C.DIR_FIGURAS.mkdir(exist_ok=True)

    T = ajustar_umbral(casos) if args.ajustar else C.UMBRAL_HU
    print(f"Umbral usado: {T} HU  |  casos: {len(casos)}")

    filas = []
    for n, (nombre, clave, m, d) in enumerate(casos):
        seg = crecimiento_regiones(d["img"], d["semilla"], umbral=T)
        val = evaluar(seg, d["consenso"], d["radiologos"])
        otsu, _ = segmentar_otsu(d["img"], d["semilla"])
        pulmon = pulmon_roi(d)
        fila = dict(**clave, grupo=m.get("grupo"), conjunto=m.get("conjunto"),
                    malignidad_mediana=m.get("malignidad_mediana"),
                    diametro_medio_mm=m.get("diametro_medio_mm"),
                    pixel_mm=d["pixel_mm_original"].item(), umbral_hu=T, **val,
                    dice_otsu=dice(otsu, d["consenso"]),
                    n_radiologos_corte=int(d["radiologos"].any(axis=(1, 2)).sum()))
        fila.update(caracteristicas_interior(d["img"], seg))
        ent, anillos = caracteristicas_entorno(d["img"], seg, pulmon)
        fila.update(ent)
        filas.append(fila)
        if n < args.figuras:
            figura_caso(d["img"], d["consenso"], d["radiologos"], seg, d["semilla"],
                        titulo=f"{nombre} | {m.get('grupo')} | Dice {val['dice']:.2f} "
                               f"(radiólogos {val['concordancia_radiologos']:.2f}) "
                               f"{'ACEPTABLE' if val['aceptable'] else 'no aceptable'}",
                        anillos=anillos, ruta=C.DIR_FIGURAS / f"05_caso_{nombre}.png")

    df = pd.DataFrame(filas)
    df.to_csv(C.DIR_RESULTADOS / "caracteristicas.csv", index=False)
    print(f"\nDice medio: {df.dice.mean():.3f} (Otsu {df.dice_otsu.mean():.3f}) | "
          f"concordancia radiólogos media: {df.concordancia_radiologos.mean():.3f}")
    print(f"Aceptables: {df.aceptable.sum()}/{len(df)} ({100 * df.aceptable.mean():.0f} %)")
    if df["grupo"].notna().any():
        print(df.groupby("grupo")[["dice", "aceptable", "diametro_eq_mm"]].mean().round(3))
    print("->", C.DIR_RESULTADOS / "caracteristicas.csv")


if __name__ == "__main__":
    main()
