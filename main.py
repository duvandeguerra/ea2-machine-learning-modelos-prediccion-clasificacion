from src.data_loader import HFDatasetLoader

DATASET_ID = "scikit-learn/credit-card-clients"

def main():
    print(f"Dataset: {DATASET_ID}")
    print(f"Enlace: https://huggingface.co/datasets/{DATASET_ID}\n")

    loader = HFDatasetLoader(DATASET_ID, split="train")
    df = loader.load()

    print("Dimensiones:", df.shape)
    print("\nTipos de datos:\n", df.dtypes.to_string())
    print("\nNulos totales:", int(df.isnull().sum().sum()))
    print("Duplicados:", int(df.duplicated().sum()))
    print("\nMuestra:\n", df.head().to_string())

    # Exploración para decidir el preprocesamiento
    for col in ["EDUCATION", "MARRIAGE", "PAY_0"]:
        print(f"\nValores de {col}:\n", df[col].value_counts().sort_index().to_string())
    print("\nDistribución del objetivo de clasificación:\n",
          df["default.payment.next.month"].value_counts(normalize=True).round(3).to_string())

    from src.preprocessing import DataPreprocessor, TARGET_CLF, PAY_COLS, BILL_COLS, PAY_AMT_COLS

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
    print("\nProporción de clases en train:\n", train[TARGET_CLF].value_counts(normalize=True).round(3).to_string())
    print("Proporción de clases en test:\n", test[TARGET_CLF].value_counts(normalize=True).round(3).to_string())

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


if __name__ == "__main__":
    main()