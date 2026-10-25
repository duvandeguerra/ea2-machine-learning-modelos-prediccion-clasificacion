from datetime import datetime
from pathlib import Path
import numpy as np 
from src.preprocessing import BILL_COLS, PAY_COLS


def _nivel_auc(auc: float) -> str:
    if auc >= 0.9:
        return "excelente"
    if auc >= 0.8:
        return "buena"
    if auc >= 0.7:
        return "aceptable"
    return "débil"


class ReportGenerator:
    """Genera un .txt con los resultados de ambos modelos a partir de la ejecución."""

    SEP = "=" * 70

    def __init__(self, dataset_id, split, forma_inicial, forma_final,
                 operaciones, n_train, n_test, reg, clf, datos):
        self.dataset_id = dataset_id
        self.split = split
        self.forma_inicial = forma_inicial
        self.forma_final = forma_final
        self.operaciones = operaciones
        self.n_train = n_train
        self.n_test = n_test
        self.reg = reg
        self.clf = clf
        self.datos = datos

    # ---------- secciones ----------
    def _encabezado(self) -> list[str]:
        return [
            self.SEP,
            "RESUMEN AUTOMÁTICO - MODELOS DE REGRESIÓN Y CLASIFICACIÓN",
            self.SEP,
            f"Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
        ]

    def _dataset(self) -> list[str]:
        return [
            "1. DATASET",
            "-" * 70,
            f"Identificador: {self.dataset_id}",
            f"Enlace: https://huggingface.co/datasets/{self.dataset_id}",
            f"Split: {self.split}",
            "Contenido: clientes de tarjetas de crédito en Taiwán (abr-sep 2005): cupo,",
            "datos demográficos, historial de pagos, facturas, pagos e incumplimiento.",
            f"Dimensiones originales: {self.forma_inicial[0]} filas x {self.forma_inicial[1]} columnas",
            "",
        ]

    def _preprocesamiento(self) -> list[str]:
        lineas = ["2. PREPROCESAMIENTO", "-" * 70]
        lineas += [f"- {op}" for op in self.operaciones]
        lineas += [
            "- En Pipeline (ajustado solo con train): StandardScaler para numéricas y",
            "  OneHotEncoder para EDUCATION y MARRIAGE.",
            f"Dimensiones finales: {self.forma_final[0]} filas x {self.forma_final[1]} columnas",
            "",
        ]
        return lineas

    def _regresion(self) -> list[str]:
        r, m = self.reg, self.reg.metrics
        mejora = (1 - m["MAE"] / m["MAE_baseline"]) * 100
        top = r.coefficients().head(8)
        return [
            "3. MODELO DE REGRESIÓN LINEAL",
            "-" * 70,
            "Objetivo: estimar el cupo de crédito asignado a un cliente según su perfil.",
            f"Variable objetivo: {r.target} (NT$)",
            f"Variables predictoras ({len(r.features)}): {', '.join(r.features)}",
            "Excluidas: default (información futura), SEX (ética), ID.",
            "Algoritmo: LinearRegression (mínimos cuadrados ordinarios).",
            f"Parámetros (predeterminados): {r.params}",
            "Codificación: OneHotEncoder drop='first' (referencia: EDUCATION=1 posgrado, MARRIAGE=1 casado).",
            f"Entrenamiento / prueba: {self.n_train} / {self.n_test} filas.",
            "",
            "Métricas (conjunto de prueba):",
            f"  MAE:  {m['MAE']:,.0f} NT$   (baseline media: {m['MAE_baseline']:,.0f} NT$)",
            f"  RMSE: {m['RMSE']:,.0f} NT$",
            f"  R2:   {m['R2']:.3f}",
            f"  Intercepto: {r.intercept:,.0f} NT$",
            "",
            "Coeficientes principales (por desviación estándar, o diferencia vs. categoría de referencia):",
            top.to_string(index=False, float_format=lambda x: f"{x:,.0f}"),
            "",
            "Interpretación:",
            f"  El modelo reduce el error absoluto medio un {mejora:.0f}% frente a predecir siempre la media",
            f"  y explica el {m['R2'] * 100:.0f}% de la variación del cupo. El RMSE es "
            f"{m['RMSE'] / m['MAE']:.2f} veces el MAE,",
            "  lo que indica errores grandes en algunos clientes (probablemente con cupos altos).",
        ] + self._interpretacion_coef_regresion() + [""]

    def _clasificacion(self) -> list[str]:
        c, m = self.clf, self.clf.metrics
        tn, fp, fn, tp = c.matriz.ravel()
        top = c.coefficients().head(8)
        return [
            "4. MODELO DE CLASIFICACIÓN",
            "-" * 70,
            "Objetivo: predecir si un cliente incumplirá el pago el próximo mes.",
            f"Variable objetivo: {c.target} (0 = {c.CLASES[0]}, 1 = {c.CLASES[1]})",
            f"Variables predictoras ({len(c.features)}): {', '.join(c.features)}",
            "Algoritmo: LogisticRegression. Elegido por ser interpretable (odds ratios),",
            "entregar probabilidades y ser la referencia estándar en clasificación binaria.",
            f"Parámetros: {c.params}",
            "class_weight='balanced' compensa el desbalance de clases; max_iter=1000 asegura",
            "convergencia; C=1.0 y regularización L2 son los valores predeterminados.",
            f"Entrenamiento / prueba: {self.n_train} / {self.n_test} filas (estratificado).",
            "",
            "Matriz de confusión (conjunto de prueba):",
            c.confusion_df().to_string(),
            "",
            "Reporte por clase:",
            c.reporte,
            f"Exactitud: {m['accuracy']:.3f}   (baseline 'nadie incumple': {m['accuracy_baseline']:.3f})",
            f"ROC AUC:   {m['roc_auc']:.3f}",
            "",
            "Coeficientes principales (odds ratio > 1 aumenta el riesgo, < 1 lo reduce):",
            top.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
            "",
            "Interpretación:",
            "  Por el desbalance de clases, se priorizan el recall y el F1 de 'Incumple' sobre la exactitud:",
            "  un modelo que nunca predice incumplimiento logra una exactitud alta pero no detecta riesgos.",
            f"  El modelo detecta {tp} de {tp + fn} incumplimientos (recall {m['recall_incumple']:.1%}) y",
            f"  no detecta {fn}. Genera {fp} falsas alarmas; de sus alertas, el "
            f"{m['precision_incumple']:.1%} son correctas.",
            f"  ROC AUC de {m['roc_auc']:.3f}: capacidad de discriminación {_nivel_auc(m['roc_auc'])}.",
            "",
        ]

    def _conclusiones(self) -> list[str]:
        mr = self.reg.metrics
        tn, fp, fn, tp = self.clf.matriz.ravel()
        top_reg = self._top_numerica(self.reg.coefficients())
        top_clf = self._top_numerica(self.clf.coefficients())
        corr_ret, bill_min, bill_max = self._correlaciones()

        return [
            "5. CONCLUSIONES Y LIMITACIONES",
            "-" * 70,
            # --- Conclusiones calculadas a partir de la ejecución ---
            f"- Variable numérica más influyente: regresión -> {top_reg}; "
            f"clasificación -> {top_clf}.",
            f"- Multicolinealidad: correlación máxima entre meses_con_retraso y PAY_x = {corr_ret:.2f};",
            f"  entre BILL_AMT de distintos meses, entre {bill_min:.2f} y {bill_max:.2f}.",
            "  Por eso sus coeficientes individuales no deben interpretarse por separado.",
            f"- Regresión: el {(1 - mr['R2']) * 100:.0f}% de la variación del cupo queda sin explicar,",
            "  atribuible a variables ausentes del dataset (ingresos, patrimonio). Además, el cupo",
            "  limita la factura, por lo que BILL_AMT refleja en parte causalidad inversa.",
            f"- Clasificación: con umbral 0.5 no se detecta el {fn / (tp + fn):.0%} de los incumplimientos",
            f"  ({fn} clientes); ajustar el umbral o probar modelos no lineales podría mejorar el recall.",
            # --- Contexto y decisiones de diseño (no dependen de los resultados) ---
            "- Datos de Taiwán (2005): la generalización a otros mercados o épocas no está garantizada.",
            "- SEX se excluyó por consideraciones éticas de no discriminación.",
        ]

    # ---------- API pública ----------
    def build(self) -> str:
        lineas = (self._encabezado() + self._dataset() + self._preprocesamiento()
                  + self._regresion() + self._clasificacion() + self._conclusiones())
        return "\n".join(lineas)

    def save(self, path: str) -> str:
        ruta = Path(path)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(self.build(), encoding="utf-8")
        return str(ruta)

    @staticmethod
    def _top_numerica(tabla) -> str:
        """Variable numérica con mayor coeficiente absoluto (la tabla ya viene ordenada)."""
        nums = tabla[tabla["variable"].str.startswith("num__")]
        return nums.iloc[0]["variable"].removeprefix("num__")

    def _correlaciones(self) -> tuple[float, float, float]:
        d = self.datos
        corr_retraso = (d[PAY_COLS + ["meses_con_retraso"]].corr()["meses_con_retraso"]
                        .drop("meses_con_retraso").abs().max())
        bill = d[BILL_COLS].corr().abs().to_numpy()
        pares = bill[np.triu_indices_from(bill, k=1)]  # solo pares distintos
        return corr_retraso, pares.min(), pares.max()

    def _interpretacion_coef_regresion(self) -> list[str]:
        coefs = self.reg.coefficients().set_index("variable")["coeficiente"]
        nombres_edu = {"2": "universidad", "3": "bachillerato", "4": "otros"}
        edu = {k.split("_")[-1]: v for k, v in coefs.items() if k.startswith("cat__EDUCATION")}
        detalle = ", ".join(f"{nombres_edu.get(k, k)} {v:,.0f}" for k, v in sorted(edu.items()))
        direccion = "menor" if all(v < 0 for v in edu.values()) else "distinto"
        pay_neg = sum(coefs[f"num__{p}"] < 0 for p in PAY_COLS)
        return [
            f"  Frente a posgrado, las demás categorías educativas tienen un cupo estimado {direccion}",
            f"  ({detalle} NT$). {pay_neg} de {len(PAY_COLS)} coeficientes PAY_x son negativos",
            "  (más retraso, menor cupo estimado).",
        ]