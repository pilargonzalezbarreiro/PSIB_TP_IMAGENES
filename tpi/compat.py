"""Compatibilidad de pylidc con versiones nuevas de numpy y configuración.

pylidc (2016) usa alias como np.int / np.bool / np.float, que numpy borró
en la 1.24. En vez de instalar un numpy viejo (que no existe para Python
3.12+), volvemos a crear esos alias antes de importar pylidc.
"""
import configparser
import os
import platform
from pathlib import Path

import numpy as np

for _alias, _tipo in (("int", int), ("bool", bool), ("float", float)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _tipo)

# pylidc usa configparser.SafeConfigParser para leer ~/.pylidcrc; ese nombre
# viejo se eliminó en Python 3.12 (era un alias de ConfigParser).
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

# pylidc también hace `import pkg_resources` (parte de setuptools) solo para
# encontrar su archivo pylidc.sqlite. Desde Python 3.12 los entornos virtuales
# nuevos ya no traen setuptools, y las versiones nuevas de setuptools quitaron
# pkg_resources. Si no está, armamos un reemplazo mínimo con lo único que usa.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import importlib.util
    import sys
    import types

    _pr = types.ModuleType("pkg_resources")

    def _resource_filename(paquete, recurso):
        carpeta = Path(importlib.util.find_spec(paquete).origin).parent
        return str(carpeta / recurso)

    _pr.resource_filename = _resource_filename
    sys.modules["pkg_resources"] = _pr


def archivo_config_pylidc() -> Path:
    nombre = "pylidc.conf" if platform.system() == "Windows" else ".pylidcrc"
    return Path.home() / nombre


def configurar_pylidc(ruta_dicom) -> Path:
    """Escribe ~/.pylidcrc con la carpeta de los DICOM (lo pide pylidc)."""
    ruta_dicom = Path(ruta_dicom).resolve()
    ruta_dicom.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": str(ruta_dicom), "warn": "True"}
    destino = archivo_config_pylidc()
    with open(destino, "w") as f:
        cfg.write(f)
    return destino


def importar_pylidc():
    import pylidc as pl  # noqa: E402  (después de los alias)
    return pl


os.environ.setdefault("PYTHONWARNINGS", "ignore")
