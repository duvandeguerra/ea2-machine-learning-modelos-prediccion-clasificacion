import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_ORIGINAL = "default.payment.next.month"
TARGET_CLF = "default"
TARGET_REG = "LIMIT_BAL"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAS = ["EDUCATION", "MARRIAGE"]


class DataPreprocessor:
    """Limpia el DataFrame con reglas fijas, lo divide y construye el transformador."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.operaciones: list[str] = []  # registro para el reporte .txt

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        filas_iniciales = len(df)

        # 1. ID: identificador sin información predictiva
        df = df.drop(columns=["ID"])
        self.operaciones.append("Se eliminó ID (identificador sin valor predictivo).")

        # 2. Duplicados reales (solo se detectan sin el ID)
        duplicados = int(df.duplicated().sum())
        df = df.drop_duplicates()
        self.operaciones.append(f"Se eliminaron {duplicados} filas duplicadas (detectadas sin ID).")

        # 3. SEX: excluido por consideraciones éticas
        df = df.drop(columns=["SEX"])
        self.operaciones.append("Se eliminó SEX: usar el género en decisiones de crédito es discriminatorio.")

        # 4. Nombre corto para el objetivo
        df = df.rename(columns={TARGET_ORIGINAL: TARGET_CLF})

        # 5. Códigos no documentados -> categoría 'otros'
        edu_raros = int(df["EDUCATION"].isin([0, 5, 6]).sum())
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        mar_raros = int((df["MARRIAGE"] == 0).sum())
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
        self.operaciones.append(
            f"EDUCATION: {edu_raros} valores 0/5/6 agrupados en 4 (otros). "
            f"MARRIAGE: {mar_raros} valores 0 agrupados en 3 (otros)."
        )

        # 6. Tipos consistentes en montos
        montos = ["LIMIT_BAL"] + BILL_COLS + PAY_AMT_COLS
        df[montos] = df[montos].astype(float)
        self.operaciones.append("Montos (LIMIT_BAL, BILL_AMT, PAY_AMT) unificados a float.")

        # 7. Enriquecimiento: meses con retraso (regla por fila, sin fuga)
        df["meses_con_retraso"] = (df[PAY_COLS] > 0).sum(axis=1)
        self.operaciones.append("Nueva variable meses_con_retraso: meses (de 6) con PAY_x > 0.")

        self.operaciones.append(f"Filas: {filas_iniciales} -> {len(df)}. Columnas finales: {df.shape[1]}.")
        return df

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[TARGET_CLF],
        )
        self.operaciones.append(
            f"División train/test {int((1 - self.test_size) * 100)}/{int(self.test_size * 100)}, "
            f"random_state={self.random_state}, estratificada por {TARGET_CLF}: "
            f"{len(train)} train, {len(test)} test."
        )
        return train, test

    @staticmethod
    def build_transformer(numericas: list[str]) -> ColumnTransformer:
        """Pasos que aprenden de los datos: se ajustan dentro del Pipeline solo con train."""
        return ColumnTransformer([
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ])