def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def fuse_detections(yolo, rcnn, threshold=0.4):
    """Match same-class detections using IoU and confidence-weighted boxes."""
    used_rcnn = set()
    fused = []
    agreements = 0

    for yd in yolo:
        best_index = -1
        best_iou = 0.0

        for i, rd in enumerate(rcnn):
            if i in used_rcnn or yd["class"] != rd["class"]:
                continue

            score = iou(yd["bbox"], rd["bbox"])
            if score > best_iou:
                best_iou = score
                best_index = i

        if best_index >= 0 and best_iou >= threshold:
            rd = rcnn[best_index]
            used_rcnn.add(best_index)
            agreements += 1

            yc = yd["confidence"]
            rc = rd["confidence"]
            total_conf = yc + rc
            box = [
                int(round((yd["bbox"][j] * yc + rd["bbox"][j] * rc) / total_conf))
                for j in range(4)
            ]

            fused.append(
                {
                    "bbox": box,
                    "class": yd["class"],
                    "confidence": (yc + rc) / 2,
                    "model": "Fused",
                    "agreement": True,
                }
            )
        else:
            item = yd.copy()
            item["agreement"] = False
            fused.append(item)

    for i, rd in enumerate(rcnn):
        if i not in used_rcnn:
            item = rd.copy()
            item["agreement"] = False
            fused.append(item)

    return fused, agreements
