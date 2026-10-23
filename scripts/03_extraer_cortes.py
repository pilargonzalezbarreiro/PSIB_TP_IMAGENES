"""Etapa 3 - Corte de trabajo de cada nódulo descargado.

Para cada nódulo: volumen en HU, máscara de consenso (≥50 % de radiólogos),
corte axial de mayor área, semilla, remuestreo a 0,7 mm y recorte de 64 mm.
Guarda un .npz por nódulo en datos/cortes/ para no volver a leer DICOM.

Uso:  python scripts/03_extraer_cortes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.compat import configurar_pylidc, importar_pylidc  # noqa: E402
from tpi.datos import extraer_corte, guardar_corte  # noqa: E402


def main():
    configurar_pylidc(C.DIR_DICOM)
    pl = importar_pylidc()
    C.DIR_CORTES.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(C.DIR_RESULTADOS / "nodulos_seleccionados.csv")
    ok = falta = 0
    for scan_id, grupo_scan in sel.groupby("scan_id"):
        scan = pl.query(pl.Scan).filter(pl.Scan.id == int(scan_id)).first()
        try:
            scan.get_path_to_dicom_files()
        except (RuntimeError, IOError):  # esta serie no se descargó
            falta += len(grupo_scan)
            continue
        nodulos = scan.cluster_annotations()
        for _, fila in grupo_scan.iterrows():
            destino = C.DIR_CORTES / f"{fila.paciente}_s{scan_id}_n{fila.nodulo}.npz"
            if destino.exists():
                ok += 1
                continue
            datos = extraer_corte(scan, nodulos[int(fila.nodulo)])
            guardar_corte(destino, datos)
            ok += 1
            print(f"  {destino.name}  corte {datos['indice_corte']}  "
                  f"radiólogos en el corte: {int(datos['radiologos'].any(axis=(1, 2)).sum())}")
    print(f"\nListos: {ok}  |  sin DICOM descargado: {falta}")


if __name__ == "__main__":
    main()
