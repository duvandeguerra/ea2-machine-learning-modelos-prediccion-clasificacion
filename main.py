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

if __name__ == "__main__":
    main()