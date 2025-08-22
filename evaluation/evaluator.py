"""
Avaliação do modelo: compatível com main.py e evaluate.py.
Aceita (model, config) OU (config=...) + load_model(...).
"""

from typing import Dict
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model


class ModelEvaluator:
    def __init__(self, model=None, config=None):
        """
        Pode receber:
          - ModelEvaluator(model, config)
          - ModelEvaluator(config=config)  # e depois chamar load_model(...)
        """
        self.model = model
        self.config = config

    # ---------- Carregamento ----------
    def load_model(self, model_path: str):
        """Carrega um modelo .keras e armazena em self.model."""
        self.model = keras_load_model(model_path)
        return self.model

    # ---------- Checagem ----------
    def _ensure_model(self):
        if self.model is None:
            raise RuntimeError(
                "ModelEvaluator: 'model' não definido. "
                "Passe 'model' no construtor ou chame load_model(...)."
            )

    # ---------- Predição ----------
    def predict(self, images_processed: np.ndarray) -> np.ndarray:
        """Retorna predições no mesmo espaço das labels de treino (ex.: 0..1)."""
        self._ensure_model()
        return self.model.predict(images_processed, verbose=0)

    # ---------- Avaliação ----------
    def evaluate_model(self, images_processed: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Executa model.evaluate e retorna um dicionário com métricas básicas.
        Assume que o modelo foi compilado com loss e MAE.
        """
        self._ensure_model()
        loss, mae = self.model.evaluate(images_processed, y_true, verbose=0)
        acc_pct = 100.0 * (1.0 - float(mae))
        err_pct = 100.0 - acc_pct
        return {
            "loss": float(loss),
            "mae": float(mae),
            "accuracy_percentage": float(acc_pct),
            "error_percentage": float(err_pct),
        }

    # ---------- Métrica simples agregada ----------
    def calculate_overall_mae(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """MAE médio por coordenada (no mesmo espaço da label de treino)."""
        return float(np.mean(np.abs(y_pred - y_true)))

    # ---------- Acesso conveniente p/ prints ----------
    def get_prediction_coordinates(self, predictions: np.ndarray, i: int):
        xmin, ymin, xmax, ymax = predictions[i].tolist()
        return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

    def get_ground_truth_coordinates(self, ground_truth: np.ndarray, i: int):
        xmin, ymin, xmax, ymax = ground_truth[i].tolist()
        return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
