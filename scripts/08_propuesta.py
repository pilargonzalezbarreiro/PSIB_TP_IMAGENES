"""Etapa 8 (plus) - Propuesta: biomarcadores robustos para tamizaje de baja dosis.

El 07 mostró dos problemas: la textura se rompe con el ruido (y con ella el
puntaje armado con dosis normal), y el ruido fabrica calcio falso. Este script
no cambia nada del pipeline: lee resultados/07_baja_dosis.csv y responde tres
preguntas, cada una con una propuesta concreta.

  1. ¿Cuánto ruido aguanta cada medida? Para cada medida, el mayor ruido
     agregado con el que sigue siendo reproducible (CCC de Lin >= 0,9 en esa
     dosis y en todas las de menos ruido). Se reporta también el ICC
     (correlación intraclase de acuerdo absoluto, la que prometía el
     anteproyecto); los dos dan casi lo mismo.
  2. Puntaje robusto por diseño: el mismo puntaje del 06 (hasta 5 medidas,
     pesos iguales, sin repetidas), pero solo puede elegir medidas que separan
     (AUC >= 0,7 en el pliegue de armado) y que son reproducibles a la dosis de uso
     (CCC >= 0,9 a dosis x1/4, del orden de la de un tamizaje). El CCC no usa las etiquetas benigno /
     sospechoso, así que filtrar por CCC no "espía" la clasificación. Se
     compara transferido (armado con la original, aplicado a baja dosis).
  3. Calcio robusto al ruido (tpi.caracteristicas.fraccion_calcio_robusta:
     mediana 3x3 + umbral + etiquetado y área mínima): ¿baja el calcio falso
     en los nódulos sin calcio y se siguen detectando los calcificados?
     Necesita la columna frac_calcio_robusto (07 corrido con la versión que
     la calcula).

Todo se hace dos veces: con todos los nódulos y solo con los sin calcio
(la pregunta clínica abierta). Ojo: el CCC es sensible a unos pocos casos
extremos (un nódulo calcificado, o una segmentación que se escapa en baja
dosis), por eso el grupo sin calcio es el más limpio para leer la
reproducibilidad.

Uso:  python scripts/08_propuesta.py
Salidas (sufijo _sincalcio para el segundo grupo):
  resultados/08_ruido_tolerable*.csv
  resultados/08_puntaje_robusto*.csv
  resultados/08_calcio_robusto.csv
  figuras/08_propuesta*.png
"""
import importlib.util
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
from tpi.puntaje import Puntaje, auc, ic_bootstrap  # noqa: E402

# reusar las funciones del 07 (el nombre empieza con número, se importa así)
_spec = importlib.util.spec_from_file_location("e07", Path(__file__).with_name("07_baja_dosis.py"))
E7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E7)

CLAVES = ["paciente", "scan_id", "nodulo"]
DOSIS = ["FBP sin ruido", "dosis ×1", "dosis ×1/4", "dosis ×1/10", "dosis ×1/20"]
# Un tamizaje de baja dosis usa del orden de 1/4 de la dosis de una tomografía
# diagnóstica: es la dosis "de uso" para decidir qué medidas son aceptables.
DOSIS_USO = "dosis ×1/4"
UMBRAL_REPRO = 0.9
NO_MEDIDAS = {"ruido_hu", "ruido_agregado_hu", "scan_id", "nodulo", "frac_calcio_robusto"}


def icc_acuerdo(x, y):
    """ICC(A,1): correlación intraclase de dos vías, acuerdo absoluto, una medición."""
    ok = np.isfinite(x) & np.isfinite(y)
    Y = np.column_stack([x[ok], y[ok]])
    n, k = Y.shape
    if n < 6:
        return np.nan
    m = Y.mean()
    msr = k * ((Y.mean(axis=1) - m) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(axis=0) - m) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + m) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


AUC_MIN = 0.7


class PuntajeRobusto(Puntaje):
    """El mismo puntaje del 06 (pesos iguales), con una regla más: de las
    medidas candidatas solo se quedan las que separan los grupos en el
    pliegue de armado (AUC >= AUC_MIN). Con pesos iguales, sumar una medida
    que casi no separa diluye a las buenas. Si ninguna llega, queda la mejor."""

    def ajustar(self, X, y):
        super().ajustar(X, y)
        buenas = self.medidas[self.medidas.auc_entrenamiento >= AUC_MIN]
        self.medidas = (buenas if len(buenas) else self.medidas.head(1)).reset_index(drop=True)
        cols = self.medidas["medida"].tolist()
        self.media, self.desvio = X[cols].mean(), X[cols].std(ddof=1).replace(0, 1)
        return self


def cv_robusto(X_arma, X_aplica, y, pacientes, k):
    pl = E7.pliegues(pacientes)
    s = np.full(len(y), np.nan)
    for f in np.unique(pl):
        tr, te = pl != f, pl == f
        s[te] = PuntajeRobusto(k).ajustar(X_arma[tr], y[tr]).puntuar(X_aplica[te])
    return s


def degenerada(x):
    """Medida casi constante (ej. fracción de aire: 0 en casi todos): su CCC no significa nada."""
    x = x[np.isfinite(x)]
    return len(x) == 0 or np.mean(x == np.median(x)) > 0.8 or np.std(x) == 0


def poblacion(df):
    car = pd.read_csv(C.DIR_RESULTADOS / "caracteristicas.csv")
    pob = car[car.grupo.isin(["benigno", "sospechoso"]) & (car.conjunto == "entrenamiento") & car.aceptable]
    pob = pob[CLAVES + ["grupo", "frac_calcio"]].rename(columns={"frac_calcio": "calcio_original"})
    return df.drop(columns=["concordancia_radiologos"], errors="ignore").merge(pob, on=CLAVES)


def tabla(df, cond):
    base = df[df.condicion == "original"].set_index(CLAVES)
    return base, df[df.condicion == cond].set_index(CLAVES).reindex(base.index)


def ruido_tolerable(df, medidas):
    ruido = df.groupby("condicion").ruido_agregado_hu.median()
    filas = []
    for c in medidas:
        fila = dict(medida=c)
        tolera = 0.0
        sigue = True
        for cond in DOSIS:
            base, otra = tabla(df, cond)
            x, y = base[c].to_numpy(float), otra[c].to_numpy(float)
            v = np.nan if degenerada(x) else E7.ccc(x, y)
            fila[f"CCC {cond}"] = v
            fila[f"ICC {cond}"] = icc_acuerdo(x, y)
            if sigue and np.isfinite(v) and v >= UMBRAL_REPRO:
                tolera = ruido[cond]
            else:
                sigue = False
        fila["ruido_tolerable_hu"] = tolera
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("ruido_tolerable_hu", ascending=False)


def puntaje_robusto(df, medidas, robustas, k=5):
    entorno = [c for c in medidas if c.startswith("anillo_") or c in E7.ENTORNO_EXTRA]
    interior = [c for c in medidas if c not in entorno]
    conjuntos = {"diámetro solo": (["diametro_eq_mm"], 1),
                 "interior (como en el 06)": (interior, k),
                 "interior robusto (propuesta)": ([c for c in interior if c in robustas], k)}
    base, _ = tabla(df, "original")
    base = base.reset_index()
    y = (base.grupo == "sospechoso").to_numpy(int)
    pac = base.paciente.to_numpy()
    filas = []
    for cond in ["original"] + DOSIS[1:]:
        _, otra = tabla(df, cond)
        otra = otra.reset_index()
        for nombre, (cols, kk) in conjuntos.items():
            cv = cv_robusto if "robusto" in nombre else E7.cv_puntaje
            s = cv(base[cols], otra[cols], y, pac, kk)
            lo, hi = ic_bootstrap(y, s, base.paciente)
            filas.append(dict(condicion=cond, puntaje=nombre, auc_transferido=auc(y, s), ic95_inf=lo, ic95_sup=hi))
    return pd.DataFrame(filas), conjuntos


def calcio(df):
    if "frac_calcio_robusto" not in df or df["frac_calcio_robusto"].isna().all():
        return None
    filas = []
    for cond in ["original"] + DOSIS[1:]:
        sub = df[df.condicion == cond]
        sin = sub[sub.calcio_original <= C.FRAC_CALCIO_MAX]
        con = sub[sub.calcio_original > C.FRAC_CALCIO_MAX]
        filas.append({
            "condicion": cond,
            "sin calcio: calcio falso medio (umbral simple)": sin.frac_calcio.mean(),
            "sin calcio: calcio falso medio (robusto)": sin.frac_calcio_robusto.mean(),
            "sin calcio: % que parece calcificado (umbral simple)": 100 * (sin.frac_calcio > C.FRAC_CALCIO_MAX).mean(),
            "sin calcio: % que parece calcificado (robusto)": 100 * (sin.frac_calcio_robusto > C.FRAC_CALCIO_MAX).mean(),
            "calcificados: % detectado (umbral simple)": 100 * (con.frac_calcio > C.FRAC_CALCIO_MAX).mean(),
            "calcificados: % detectado (robusto)": 100 * (con.frac_calcio_robusto > C.FRAC_CALCIO_MAX).mean(),
            "n sin / con calcio": f"{len(sin)} / {len(con)}",
        })
    return pd.DataFrame(filas)


def analizar(df, sufijo, titulo):
    n = df.drop_duplicates(CLAVES).grupo.value_counts().to_dict()
    print(f"\n################ {titulo}: {sum(n.values())} nódulos {n} ################")
    medidas = [c for c in df.select_dtypes("number").columns
               if c not in META and c not in NO_MEDIDAS and c != "calcio_original"]

    # 1
    rt = ruido_tolerable(df, medidas)
    rt.to_csv(C.DIR_RESULTADOS / f"08_ruido_tolerable{sufijo}.csv", index=False)
    print(f"\n== 1. Cuánto ruido aguanta cada medida (CCC >= {UMBRAL_REPRO}); las 15 más robustas ==")
    cols = ["medida", "ruido_tolerable_hu", f"CCC {DOSIS_USO}", f"ICC {DOSIS_USO}", "CCC dosis ×1/10"]
    print(rt[cols].head(15).round(2).to_string(index=False))

    # 2
    robustas = set(rt.loc[rt[f"CCC {DOSIS_USO}"] >= UMBRAL_REPRO, "medida"])
    pr, conjuntos = puntaje_robusto(df, medidas, robustas)
    pr.to_csv(C.DIR_RESULTADOS / f"08_puntaje_robusto{sufijo}.csv", index=False)
    print(f"\n== 2. Puntaje robusto: candidatas = interior con CCC >= {UMBRAL_REPRO} a {DOSIS_USO} (y AUC >= {AUC_MIN} al armar) ==")
    print("   ", ", ".join(conjuntos["interior robusto (propuesta)"][0]))
    print("   AUC transferido (armado con la original, aplicado a cada dosis):")
    tab = pr.pivot(index="condicion", columns="puntaje", values="auc_transferido")
    print(tab.reindex(["original"] + DOSIS[1:])[list(conjuntos)].round(3).to_string())
    return rt, pr


def main():
    df = poblacion(pd.read_csv(C.DIR_RESULTADOS / "07_baja_dosis.csv"))
    rt, pr = analizar(df, "", "TODOS")
    sin = df[df.calcio_original <= C.FRAC_CALCIO_MAX]
    rt2, pr2 = analizar(sin, "_sincalcio", "SIN CALCIO")

    # 3
    ca = calcio(df)
    if ca is None:
        print("\n== 3. Calcio robusto: falta la columna frac_calcio_robusto -> correr de nuevo el 07 ==")
    else:
        ca.to_csv(C.DIR_RESULTADOS / "08_calcio_robusto.csv", index=False)
        print("\n== 3. Calcio robusto al ruido ==")
        print(ca.set_index("condicion").T.round(3).to_string())

    figura(df, rt, pr, ca, "")
    figura(sin, rt2, pr2, None, "_sincalcio")


def figura(df, rt, pr, ca, sufijo):
    ruido = df.groupby("condicion").ruido_agregado_hu.median()
    paneles = 3 if ca is not None else 2
    fig, ax = plt.subplots(1, paneles, figsize=(5.2 * paneles, 5))
    sel = pd.concat([rt.head(10), rt.tail(10)]).drop_duplicates("medida").iloc[::-1]
    ax[0].barh(sel.medida, sel.ruido_tolerable_hu, color="#4a7ab5")
    ax[0].set_xlabel("ruido agregado tolerable (HU)")
    ax[0].tick_params(axis="y", labelsize=6)
    ax[0].set_title(f"1. Ruido con el que cada medida\nsigue siendo reproducible (CCC ≥ {UMBRAL_REPRO})")
    orden = ["original"] + DOSIS[1:]
    x = [0 if c == "original" else ruido[c] for c in orden]
    for nombre, sub in pr.groupby("puntaje", sort=False):
        sub = sub.set_index("condicion").reindex(orden)
        ax[1].errorbar(x, sub.auc_transferido, yerr=[sub.auc_transferido - sub.ic95_inf,
                       sub.ic95_sup - sub.auc_transferido], marker="o", capsize=3, label=nombre)
    ax[1].axhline(0.5, color="k", lw=0.8)
    ax[1].set_xlabel("ruido agregado (HU)")
    ax[1].set_ylabel("AUC transferido")
    ax[1].set_title("2. Puntaje armado con la original,\naplicado a cada dosis")
    ax[1].legend(fontsize=7, loc="lower left")
    if ca is not None:
        xs = [0 if c == "original" else ruido[c] for c in ca.condicion]
        ax[2].plot(xs, ca["sin calcio: % que parece calcificado (umbral simple)"], "o-", color="#c0504d",
                   label="sin calcio que parecen calcificados (umbral simple)")
        ax[2].plot(xs, ca["sin calcio: % que parece calcificado (robusto)"], "o-", color="#4a7ab5",
                   label="sin calcio que parecen calcificados (robusto)")
        ax[2].plot(xs, ca["calcificados: % detectado (robusto)"], "s--", color="#4a7ab5", alpha=0.6,
                   label="calcificados detectados (robusto)")
        ax[2].set_xlabel("ruido agregado (HU)")
        ax[2].set_ylabel("% de nódulos")
        ax[2].set_ylim(-3, 103)
        ax[2].set_title("3. Calcio robusto al ruido")
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    ruta = C.DIR_FIGURAS / f"08_propuesta{sufijo}.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("->", ruta)


if __name__ == "__main__":
    main()
