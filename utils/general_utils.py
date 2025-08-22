"""
Utils gerais: (des)normalização de bounding boxes e conversões auxiliares.
"""

from typing import Iterable, Tuple

def denormalize_bbox_xyxy(
    bbox: Iterable[float],
    img_w: int,
    img_h: int,
    *,
    was_32_scale: bool = False,
    was_normalized01: bool = True
) -> Tuple[float, float, float, float]:
    """
    Converte [xmin, ymin, xmax, ymax] para pixels da imagem (img_w x img_h).

    - Se was_normalized01=True  → bbox está em 0..1 (padrão recomendado).
    - Se was_32_scale=True      → bbox está em coordenadas 0..32.

    Use exatamente UMA das duas convenções acima. Se as duas forem True,
    a prioridade é was_32_scale (0..32).
    """
    xmin, ymin, xmax, ymax = bbox

    if was_32_scale:
        sx = img_w / 32.0
        sy = img_h / 32.0
        return xmin * sx, ymin * sy, xmax * sx, ymax * sy

    if was_normalized01:
        return xmin * img_w, ymin * img_h, xmax * img_w, ymax * img_h

    # Já está em pixels
    return float(xmin), float(ymin), float(xmax), float(ymax)


def bbox_xyxy_to_circle(bbox_xyxy):
    """
    Converte [xmin, ymin, xmax, ymax] → (cx, cy, r) para desenhar círculo que cobre o bbox.
    """
    xmin, ymin, xmax, ymax = bbox_xyxy
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    r = max(xmax - xmin, ymax - ymin) / 2.0
    return cx, cy, r
