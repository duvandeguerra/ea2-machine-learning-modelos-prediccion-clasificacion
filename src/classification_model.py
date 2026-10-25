import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    BILL_COLS, CATEGORICAS, PAY_AMT_COLS, PAY_COLS, TARGET_CLF, DataPreprocessor,
)


class ClassificationModel:
    """Regresión logística para predecir el incumplimiento de pago (default)."""

    NUMERICAS = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS + ["meses_con_retraso"]
    CLASES = ["No incumple", "Incumple"]

    def __init__(self, random_state: int = 42):
        self.target = TARGET_CLF
        self.features = self.NUMERICAS + CATEGORICAS
        self.pipeline = Pipeline([
            ("prep", DataPreprocessor.build_transformer(self.NUMERICAS)),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_state,
            )),
        ])
        self.metrics: dict[str, float] = {}
        self.matriz: np.ndarray | None = None
        self.reporte: str = ""

    def train(self, train: pd.DataFrame) -> None:
        self.pipeline.fit(train[self.features], train[self.target])

    def evaluate(self, test: pd.DataFrame) -> dict[str, float]:
        y_real = test[self.target]
        y_pred = self.pipeline.predict(test[self.features])
        y_prob = self.pipeline.predict_proba(test[self.features])[:, 1]

        self.matriz = confusion_matrix(y_real, y_pred)
        self.reporte = classification_report(y_real, y_pred, target_names=self.CLASES, digits=3)
        self.metrics = {
            "accuracy": accuracy_score(y_real, y_pred),
            "accuracy_baseline": float((y_real == 0).mean()),  # predecir siempre "no incumple"
            "precision_incumple": precision_score(y_real, y_pred),
            "recall_incumple": recall_score(y_real, y_pred),
            "f1_incumple": f1_score(y_real, y_pred),
            "f1_macro": f1_score(y_real, y_pred, average="macro"),
            "roc_auc": roc_auc_score(y_real, y_prob),
        }
        return self.metrics

    def confusion_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.matriz,
            index=[f"Real: {c}" for c in self.CLASES],
            columns=[f"Pred: {c}" for c in self.CLASES],
        )

    def coefficients(self) -> pd.DataFrame:
        nombres = self.pipeline.named_steps["prep"].get_feature_names_out()
        coefs = self.pipeline.named_steps["model"].coef_[0]
        tabla = pd.DataFrame({
            "variable": nombres,
            "coeficiente": coefs,
            "odds_ratio": np.exp(coefs),
        })
        return tabla.sort_values("coeficiente", key=abs, ascending=False).reset_index(drop=True)

    @property
    def params(self) -> dict:
        return self.pipeline.named_steps["model"].get_params()