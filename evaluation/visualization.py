"""
Visualização de resultados: garante que as predições/labels sejam desnormalizadas
para o TAMANHO REAL da imagem exibida antes de desenhar.
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from utils.general_utils import denormalize_bbox_xyxy, bbox_xyxy_to_circle


class ResultVisualizer:
    def __init__(self, config):
        self.config = config
        self._fig = None
        self._ax = None

        # defina sua convenção aqui:
        # Se durante o TREINO você usou coordenadas 0..1:
        self._was_normalized01 = True
        self._was_32_scale = False

        # Se durante o TREINO você usou coordenadas 0..32 (após redimensionar para 32x32),
        # troque para:
        # self._was_normalized01 = False
        # self._was_32_scale = True

    # ---------- Interface ----------
    def plot_multiple_predictions(
        self,
        images_original: List[np.ndarray],
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        num_images: int = 12,
        draw_circle: bool = False,
    ):
        n = min(num_images, len(images_original))
        cols = 4
        rows = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        for i in range(n):
            img = images_original[i]
            h, w = img.shape[:2]

            pred_xyxy = denormalize_bbox_xyxy(
                predictions[i], w, h, was_32_scale=self._was_32_scale, was_normalized01=self._was_normalized01
            )
            gt_xyxy = denormalize_bbox_xyxy(
                ground_truth[i], w, h, was_32_scale=self._was_32_scale, was_normalized01=self._was_normalized01
            )

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.axis("off")

            # GT em verde
            self._draw_bbox(ax, gt_xyxy, edgecolor="g", label="GT")

            # Pred em vermelho
            self._draw_bbox(ax, pred_xyxy, edgecolor="r", label="Pred")

            if draw_circle:
                cx, cy, r = bbox_xyxy_to_circle(pred_xyxy)
                circ = Circle((cx, cy), r, fill=False, linewidth=1.4, edgecolor="r", alpha=0.6)
                ax.add_patch(circ)

        plt.tight_layout()
        self._fig = fig

    def plot_prediction_vs_ground_truth(
        self,
        image_original: np.ndarray,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        image_idx: int = 0,
        draw_circle: bool = False,
    ):
        img = image_original
        h, w = img.shape[:2]

        pred_xyxy = denormalize_bbox_xyxy(
            prediction, w, h, was_32_scale=self._was_32_scale, was_normalized01=self._was_normalized01
        )
        gt_xyxy = denormalize_bbox_xyxy(
            ground_truth, w, h, was_32_scale=self._was_32_scale, was_normalized01=self._was_normalized01
        )

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img)
        ax.axis("off")

        self._draw_bbox(ax, gt_xyxy, edgecolor="g", label=f"GT {image_idx}")
        self._draw_bbox(ax, pred_xyxy, edgecolor="r", label=f"Pred {image_idx}")

        if draw_circle:
            cx, cy, r = bbox_xyxy_to_circle(pred_xyxy)
            circ = Circle((cx, cy), r, fill=False, linewidth=1.8, edgecolor="r", alpha=0.6)
            ax.add_patch(circ)

        self._fig, self._ax = fig, ax

    def print_coordinates(self, coords, label="BBox", idx=None):
        xmin, ymin, xmax, ymax = coords
        prefix = f"[{idx}] " if idx is not None else ""
        print(f"{prefix}{label}: xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")

    def save_visualization(self, path: str = "predictions.png"):
        if self._fig is not None:
            self._fig.savefig(path, bbox_inches="tight", dpi=180)
            print(f"Figura salva em: {path}")

    # ---------- Internos ----------
    @staticmethod
    def _draw_bbox(ax, xyxy, *, edgecolor="r", label=None):
        xmin, ymin, xmax, ymax = xyxy
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        rect = Rectangle((xmin, ymin), w, h, fill=False, linewidth=1.8, edgecolor=edgecolor, alpha=0.8)
        ax.add_patch(rect)
        if label:
            ax.text(
                xmin,
                max(0, ymin - 3),
                label,
                color=edgecolor,
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.3, pad=1),
            )
