"""Etapa 2 - Descarga SOLO las series seleccionadas (no los 133 GB).

Usa idc-index (Imaging Data Commons, espejo oficial de TCIA en la nube).
Las carpetas quedan como  datos/LIDC-IDRI/<paciente>/<study_uid>/<series_uid>/
que es justo la estructura que pylidc busca primero.

Cada serie pesa ~100-250 MB. Para los avances alcanza con --limite 15.

Uso:  python scripts/02_descargar_dicom.py [--limite 15]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tpi import config as C  # noqa: E402
from tpi.compat import configurar_pylidc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0, help="máximo de series (0 = todas)")
    ap.add_argument("--dry-run", action="store_true", help="solo mostrar tamaño estimado")
    args = ap.parse_args()

    sel = pd.read_csv(C.DIR_RESULTADOS / "nodulos_seleccionados.csv")
    series = sel.drop_duplicates("series_uid")
    if args.limite:
        # Alternar grupos para que el subconjunto chico tenga de los dos.
        series = (series.assign(_r=series.groupby("grupo").cumcount())
                  .sort_values(["_r", "grupo"]).head(args.limite))
    uids = series["series_uid"].tolist()
    print(f"{len(uids)} series de {series.paciente.nunique()} pacientes")

    ruta = configurar_pylidc(C.DIR_DICOM)
    print("pylidc configurado en", ruta)

    from idc_index import IDCClient
    cliente = IDCClient()
    cliente.download_from_selection(
        downloadDir=str(C.DIR_DICOM), seriesInstanceUID=uids,
        dirTemplate="%PatientID/%StudyInstanceUID/%SeriesInstanceUID",
        dry_run=args.dry_run, quiet=False,
    )


if __name__ == "__main__":
    main()
