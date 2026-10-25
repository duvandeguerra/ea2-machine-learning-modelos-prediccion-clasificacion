import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    BILL_COLS, CATEGORICAS, PAY_AMT_COLS, PAY_COLS, TARGET_REG, DataPreprocessor,
)


class RegressionModel:
    """Regresión lineal para predecir el cupo de crédito (LIMIT_BAL)."""

    NUMERICAS = ["AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS + ["meses_con_retraso"]

    def __init__(self):
        self.target = TARGET_REG
        self.features = self.NUMERICAS + CATEGORICAS
        self.pipeline = Pipeline([
            ("prep", DataPreprocessor.build_transformer(self.NUMERICAS, drop_first=True)),
            ("model", LinearRegression()),  # parámetros por defecto (fit_intercept=True)
        ])
        self.metrics: dict[str, float] = {}
        self._media_train: float | None = None

    def train(self, train: pd.DataFrame) -> None:
        self.pipeline.fit(train[self.features], train[self.target])
        self._media_train = float(train[self.target].mean())

    def evaluate(self, test: pd.DataFrame) -> dict[str, float]:
        y_real = test[self.target]
        y_pred = self.pipeline.predict(test[self.features])
        y_base = np.full(len(y_real), self._media_train)  # modelo ingenuo: siempre la media

        self.metrics = {
            "MAE": mean_absolute_error(y_real, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
            "R2": r2_score(y_real, y_pred),
            "MAE_baseline": mean_absolute_error(y_real, y_base),
        }
        return self.metrics

    def coefficients(self) -> pd.DataFrame:
        nombres = self.pipeline.named_steps["prep"].get_feature_names_out()
        coefs = self.pipeline.named_steps["model"].coef_
        tabla = pd.DataFrame({"variable": nombres, "coeficiente": coefs})
        return tabla.sort_values("coeficiente", key=abs, ascending=False).reset_index(drop=True)

    @property
    def intercept(self) -> float:
        return float(self.pipeline.named_steps["model"].intercept_)

    @property
    def params(self) -> dict:
        return self.pipeline.named_steps["model"].get_params()