# utils/postprocess.py

import numpy as np
import cv2


def postprocess_mask(
    prob_map,
    percentile=98,
    min_area=1500,
    kernel_size=7
):
    """
    CASIA-tuned postprocessing:
    - Adaptive threshold using percentile
    - Morphological closing to merge regions
    - Remove small components
    - Keep only meaningful manipulation regions
    """

    # --- 1. Adaptive threshold ---
    thresh = np.percentile(prob_map, percentile)
    bin_mask = (prob_map >= thresh).astype(np.uint8) * 255

    # --- 2. Morphological closing (merge nearby regions) ---
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    bin_mask = cv2.morphologyEx(
        bin_mask, cv2.MORPH_CLOSE, kernel
    )

    # --- 3. Connected components ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    final_mask = np.zeros_like(bin_mask)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final_mask[labels == i] = 255

    return final_mask
