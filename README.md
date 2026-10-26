# EA2 – Modelos de Predicción y Clasificación

Proyecto de Machine Learning que, a partir de un dataset de Hugging Face, entrena una
**regresión lineal** para estimar el cupo de crédito de un cliente y un **clasificador**
para predecir si incumplirá su próximo pago. Todo el flujo se ejecuta automáticamente
con GitHub Actions.

## Integrantes

- Duvan Quintero Guerra (trabajo individual)

## Dataset

| | |
|---|---|
| **Identificador** | `scikit-learn/credit-card-clients` |
| **Enlace** | https://huggingface.co/datasets/scikit-learn/credit-card-clients |
| **Split** | `train` (único disponible; archivo `UCI_Credit_Card.csv`) |
| **Tamaño** | 30.000 clientes × 25 columnas |
| **Contenido** | Clientes de tarjetas de crédito en Taiwán (abril–septiembre 2005) |

**Columnas relevantes:** `LIMIT_BAL` (cupo), `EDUCATION`, `MARRIAGE`, `AGE`,
`PAY_0`…`PAY_6` (meses de retraso), `BILL_AMT1`…`6` (facturas), `PAY_AMT1`…`6` (pagos)
y `default.payment.next.month` (incumplimiento: 1 = sí, 0 = no).

## Problemas planteados

| | Regresión lineal | Clasificación |
|---|---|---|
| **Pregunta** | ¿Qué cupo tiene un cliente según su perfil y comportamiento de pago? | ¿Incumplirá el pago el próximo mes? |
| **Objetivo** | `LIMIT_BAL` (numérica continua, NT$) | `default` (0 = no incumple, 1 = incumple) |
| **Algoritmo** | `LinearRegression` (parámetros por defecto) | `LogisticRegression` (`class_weight="balanced"`, `max_iter=1000`) |
| **Excluidas** | `default` (información futura), `SEX`, `ID` | `SEX`, `ID` |

**¿Por qué regresión logística?** Es interpretable (cada coeficiente se traduce en un
odds ratio que indica si la variable aumenta o reduce el riesgo), entrega probabilidades
y es la referencia estándar en clasificación binaria. En crédito, explicar por qué se
considera riesgoso a un cliente es tan importante como acertar.

**¿Por qué se excluyó `SEX`?** Usar el género para decisiones de crédito es
discriminatorio y está prohibido en muchas legislaciones.

## Preprocesamiento

| Operación | Problema que resuelve |
|---|---|
| Eliminar `ID` | Identificador sin valor predictivo; además ocultaba duplicados |
| Eliminar 35 duplicados (detectados sin `ID`) | Filas repetidas sesgan el entrenamiento y la evaluación |
| Agrupar `EDUCATION` 0/5/6 → 4 y `MARRIAGE` 0 → 3 | Códigos no documentados en la ficha del dataset |
| Unificar montos a `float` | Tipos inconsistentes (`int64` / `float64`) entre columnas equivalentes |
| Crear `meses_con_retraso` | Resume el historial de pagos en una variable interpretable |
| División 80/20 estratificada (`random_state=42`) | Evaluar con datos no vistos manteniendo la proporción de clases (77,9% / 22,1%) |
| `StandardScaler` + `OneHotEncoder` dentro de un `Pipeline` | Se ajustan **solo con train** para evitar fuga de información |

No había valores nulos, por lo que no fue necesaria la imputación.
**Evidencia de que no hubo fuga:** tras escalar, `LIMIT_BAL` tiene media 0,000 y
desviación 1,000 en train, pero 0,007 y 0,996 en test (el escalador nunca vio test).

## Resultados principales

**Regresión:** MAE de 76.651 NT$ frente a 104.827 NT$ del baseline (predecir la media),
es decir, un 27% menos de error. R² = 0,391.

**Clasificación:** detecta 752 de 1.326 incumplimientos (recall 56,7%). ROC AUC = 0,749.
La exactitud (0,765) es menor que la del baseline "nadie incumple" (0,779), pero ese
baseline no detecta ningún incumplimiento; por el desbalance de clases se priorizan
el recall y el F1 de la clase "Incumple".

El análisis completo (coeficientes, odds ratios, matriz de confusión, limitaciones)
está en el resumen generado automáticamente.

## Estructura del proyecto

```
├── .github/workflows/pipeline.yml   # GitHub Actions
├── src/
│   ├── data_loader.py               # Carga desde Hugging Face → DataFrame
│   ├── preprocessing.py             # Limpieza, división y transformadores
│   ├── regression_model.py          # Regresión lineal + métricas + coeficientes
│   ├── classification_model.py      # Regresión logística + matriz de confusión
│   └── report.py                    # Genera el resumen .txt
├── main.py                          # Orquesta el flujo completo
├── outputs/resumen_modelos.txt      # Resumen generado
└── requirements.txt
```

## Cómo ejecutar

**Localmente:**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python main.py
```

El resumen se genera en `outputs/resumen_modelos.txt`.

**Automáticamente:** cada `push` a `main` ejecuta el workflow de GitHub Actions, que
instala dependencias, descarga el dataset, preprocesa, entrena y evalúa ambos modelos,
y publica el resumen como artefacto descargable `resumen-modelos`.

## Evidencias

| Evidencia | Ubicación |
|---|---|
| Carga y estructura del DataFrame | Log del paso "Ejecutar flujo completo" en Actions |
| Preprocesamiento antes/después | Log de Actions y sección 2 del resumen |
| Regresión: métricas, coeficientes, interpretación | Sección 3 de `outputs/resumen_modelos.txt` |
| Clasificación: matriz de confusión, métricas, interpretación | Sección 4 de `outputs/resumen_modelos.txt` |
| Conclusiones y limitaciones | Sección 5 de `outputs/resumen_modelos.txt` |
| Ejecución exitosa | <pega aquí el enlace a tu ejecución verde de Actions> |
| Artefacto descargable | Sección **Artifacts** de la ejecución anterior |