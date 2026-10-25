from src.data_loader import HFDatasetLoader
from src.preprocessing import DataPreprocessor, TARGET_CLF, PAY_COLS, BILL_COLS, PAY_AMT_COLS
from src.regression_model import RegressionModel
from src.classification_model import ClassificationModel
from src.report import ReportGenerator

DATASET_ID = "scikit-learn/credit-card-clients"


def main():
    # ---------- CARGA ----------
    print(f"Dataset: {DATASET_ID}")
    print(f"Enlace: https://huggingface.co/datasets/{DATASET_ID}\n")

    loader = HFDatasetLoader(DATASET_ID, split="train")
    df = loader.load()

    print("Dimensiones:", df.shape)
    print("\nTipos de datos:\n", df.dtypes.to_string())
    print("\nNulos totales:", int(df.isnull().sum().sum()))
    print("\nMuestra:\n", df.head().to_string())

    for col in ["EDUCATION", "MARRIAGE", "PAY_0"]:
        print(f"\nValores de {col}:\n", df[col].value_counts().sort_index().to_string())
    print("\nDistribución del objetivo de clasificación:\n",
          df["default.payment.next.month"].value_counts(normalize=True).round(3).to_string())

    # ---------- PREPROCESAMIENTO ----------
    print("\n" + "=" * 60 + "\nPREPROCESAMIENTO\n" + "=" * 60)
    print("ANTES:", df.shape)
    print("Duplicados sin ID:", int(df.drop(columns=["ID"]).duplicated().sum()))

    prep = DataPreprocessor(test_size=0.2, random_state=42)
    df_limpio = prep.clean(df)
    train, test = prep.split(df_limpio)

    print("\nDESPUÉS:", df_limpio.shape)
    print("\nEDUCATION:\n", df_limpio["EDUCATION"].value_counts().sort_index().to_string())
    print("\nMARRIAGE:\n", df_limpio["MARRIAGE"].value_counts().sort_index().to_string())
    print("\nProporción de clases en train:\n",
          train[TARGET_CLF].value_counts(normalize=True).round(3).to_string())
    print("Proporción de clases en test:\n",
          test[TARGET_CLF].value_counts(normalize=True).round(3).to_string())

    # Evidencia de escalado sin fuga: se ajusta con train y se aplica a ambos
    numericas = ["AGE", "LIMIT_BAL"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS + ["meses_con_retraso"]
    transformer = DataPreprocessor.build_transformer(numericas)
    X_train_t = transformer.fit_transform(train)
    X_test_t = transformer.transform(test)
    print("\nEscalado de LIMIT_BAL (columna 1):")
    print(f"  train -> media {X_train_t[:, 1].mean():.3f}, desv {X_train_t[:, 1].std():.3f}")
    print(f"  test  -> media {X_test_t[:, 1].mean():.3f}, desv {X_test_t[:, 1].std():.3f}")

    print("\nRegistro de operaciones:")
    for op in prep.operaciones:
        print(" -", op)

    # ---------- REGRESIÓN LINEAL ----------
    print("\n" + "=" * 60 + "\nREGRESIÓN LINEAL: predecir LIMIT_BAL\n" + "=" * 60)
    reg = RegressionModel()
    reg.train(train)
    m = reg.evaluate(test)

    print("Parámetros:", reg.params)
    print(f"\nMAE:  {m['MAE']:,.0f} NT$   (baseline media: {m['MAE_baseline']:,.0f} NT$)")
    print(f"RMSE: {m['RMSE']:,.0f} NT$")
    print(f"R²:   {m['R2']:.3f}")
    print(f"\nIntercepto: {reg.intercept:,.0f} NT$")
    print("\nCoeficientes (ordenados por magnitud):")
    print(reg.coefficients().to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # ---------- CLASIFICACIÓN ----------
    print("\n" + "=" * 60 + "\nCLASIFICACIÓN: predecir incumplimiento (default)\n" + "=" * 60)
    clf = ClassificationModel(random_state=42)
    clf.train(train)
    mc = clf.evaluate(test)

    print("Parámetros:", clf.params)
    print("\nMatriz de confusión:\n", clf.confusion_df().to_string())
    print("\nReporte por clase:\n", clf.reporte)
    print(f"Exactitud: {mc['accuracy']:.3f}  (baseline 'nadie incumple': {mc['accuracy_baseline']:.3f})")
    print(f"ROC AUC:   {mc['roc_auc']:.3f}")
    print("\nCoeficientes (top 10 por magnitud):")
    print(clf.coefficients().head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------- RESUMEN .TXT ----------
    reporte = ReportGenerator(
        dataset_id=DATASET_ID,
        split="train",
        forma_inicial=df.shape,
        forma_final=df_limpio.shape,
        operaciones=prep.operaciones,
        n_train=len(train),
        n_test=len(test),
        reg=reg,
        clf=clf,
        datos=train,
    )
    ruta = reporte.save("outputs/resumen_modelos.txt")
    print(f"\nResumen guardado en: {ruta}")

if __name__ == "__main__":
    main()