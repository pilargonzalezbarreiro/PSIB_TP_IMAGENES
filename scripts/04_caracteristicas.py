"""Etapa 4 - Análisis, segmentación, validación y extracción de características.

0. Análisis (paso 1 del pipeline de la materia): histogramas de las regiones
   para justificar el método de segmentación.

1. (opcional, --ajustar) barre el umbral T en el conjunto de ENTRENAMIENTO
   y elige el que maximiza el Dice medio contra el consenso.
2. Segmenta cada nódulo desde la semilla (y con Otsu, como comparación).
3. Valida contra el consenso con el umbral de aceptabilidad.
4. Extrae medidas del interior y del entorno.

Salidas:
  resultados/caracteristicas.csv   una fila por nódulo
  resultados/04_ajuste_umbral.csv  (si --ajustar)
  figuras/04_caso_*.png            algunos casos con contornos y anillos

Uso:  python scripts/04_caracteristicas.py [--ajustar] [--carpeta datos/cortes]
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
from tpi.segmentacion import region_permitida, segmentar_nodulo, segmentar_otsu  # noqa: E402
from tpi.validacion import dice, evaluar  # noqa: E402
from tpi.visualizacion import figura_analisis, figura_caso  # noqa: E402

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


# Primera corrida con 217 nódulos (20/30/40 mm): el Dice bajó al subir el radio
# (0,837 / 0,826 / 0,812) y el mejor quedó en el borde de la grilla, así que se
# agregan radios más chicos. Un disco grande rellena también la concavidad del
# mediastino y del hilio, y por ahí se escapan los nódulos pegados al mediastino.
RADIOS_CIERRE = (10.0, 15.0, 20.0, 30.0)


def ajustar_umbral(casos):
    """Barre el umbral T y el radio del cierre del pulmón en ENTRENAMIENTO y
    se queda con la combinación de mayor Dice medio contra el consenso."""
    filas = []
    entren = [c for c in casos if c[2].get("conjunto", "entrenamiento") == "entrenamiento"]
    for r in RADIOS_CIERRE:
        for T in range(-600, -275, 25):
            ds = [dice(segmentar_nodulo(d, umbral=T, radio_cierre_mm=r), d["consenso"])
                  for *_, d in entren]
            filas.append(dict(radio_cierre_mm=r, umbral_hu=T, dice_medio=np.mean(ds),
                              dice_mediana=np.median(ds), n=len(ds)))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(C.DIR_RESULTADOS / "04_ajuste_umbral.csv", index=False)
    print(tabla.pivot(index="umbral_hu", columns="radio_cierre_mm", values="dice_medio").round(3)
          .rename(columns=lambda r: f"Dice medio, cierre {r:g} mm").to_string())
    mejor = tabla.loc[tabla.dice_medio.idxmax()]
    return int(mejor["umbral_hu"]), float(mejor["radio_cierre_mm"])


def parametros_guardados():
    ajuste = C.DIR_RESULTADOS / "04_ajuste_umbral.csv"
    if ajuste.exists():
        tabla = pd.read_csv(ajuste)
        if "radio_cierre_mm" in tabla:
            mejor = tabla.loc[tabla.dice_medio.idxmax()]
            return int(mejor["umbral_hu"]), float(mejor["radio_cierre_mm"])
    return C.UMBRAL_HU, C.RADIO_CIERRE_PULMON_MM


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

    figura_analisis([d for *_, d in casos], C.DIR_FIGURAS / "04_analisis_histogramas.png")
    T, R = ajustar_umbral(casos) if args.ajustar else parametros_guardados()
    print(f"Umbral usado: {T} HU  |  cierre del pulmón: {R:g} mm  |  casos: {len(casos)}")

    filas, guardar = [], []
    for n, (nombre, clave, m, d) in enumerate(casos):
        seg, cruda = segmentar_nodulo(d, umbral=T, devolver_cruda=True, radio_cierre_mm=R)
        val = evaluar(seg, d["consenso"], d["radiologos"])
        otsu, _ = segmentar_otsu(d["img"], d["semilla"])
        pulmon = pulmon_roi(d)
        fila = dict(**clave, grupo=m.get("grupo"), conjunto=m.get("conjunto"),
                    malignidad_mediana=m.get("malignidad_mediana"),
                    diametro_medio_mm=m.get("diametro_medio_mm"),
                    pixel_mm=d["pixel_mm_original"].item(), umbral_hu=T, radio_cierre_mm=R,
                    pulmon_restringe=region_permitida(d, R) is not None, **val,
                    dice_otsu=dice(otsu, d["consenso"]),
                    n_radiologos_corte=int(d["radiologos"].any(axis=(1, 2)).sum()))
        fila.update(caracteristicas_interior(d["img"], seg, mascara_cruda=cruda))
        ent, anillos = caracteristicas_entorno(d["img"], seg, pulmon)
        fila.update(ent)
        filas.append(fila)
        guardar.append((nombre, m, d, seg, val, anillos))

    df = pd.DataFrame(filas)
    # Figuras: los primeros casos y los 6 peores (para ver CÓMO falla)
    peores = set(df.nsmallest(6, "dice").index)
    for n, (nombre, m, d, seg, val, anillos) in enumerate(guardar):
        if n < args.figuras or n in peores:
            pref = "04_peor_" if n in peores else "04_caso_"
            figura_caso(d["img"], d["consenso"], d["radiologos"], seg, d["semilla"],
                        titulo=f"{nombre} | {m.get('grupo')} | Dice {val['dice']:.2f} "
                               f"(radiólogos {val['concordancia_radiologos']:.2f}) "
                               f"{'ACEPTABLE' if val['aceptable'] else 'no aceptable'}",
                        anillos=anillos, ruta=C.DIR_FIGURAS / f"{pref}{nombre}.png")
    df.to_csv(C.DIR_RESULTADOS / "caracteristicas.csv", index=False)
    print(f"\nDice medio: {df.dice.mean():.3f}, mediana {df.dice.median():.3f} (Otsu {df.dice_otsu.mean():.3f}) | "
          f"concordancia radiólogos media: {df.concordancia_radiologos.mean():.3f}")
    print(f"Aceptables: {df.aceptable.sum()}/{len(df)} ({100 * df.aceptable.mean():.0f} %) | "
          f"Dice < 0,3: {(df.dice < 0.3).sum()} | sin restricción de pulmón: {(~df.pulmon_restringe).sum()}")
    if df["grupo"].notna().any():
        print(df.groupby("grupo")[["dice", "aceptable", "diametro_eq_mm"]].mean().round(3))
    print("->", C.DIR_RESULTADOS / "caracteristicas.csv")


if __name__ == "__main__":
    main()
