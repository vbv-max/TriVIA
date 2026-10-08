from .result_fusion import iou


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class SimpleTracker:
    """Lightweight IoU tracker with a center-distance fallback."""

    def __init__(self, match_iou=0.30, max_missing=8):
        self.match_iou = float(match_iou)
        self.max_missing = int(max_missing)
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        results = []
        matched_ids = set()

        ordered = sorted(
            detections,
            key=lambda d: d.get("confidence", 0.0),
            reverse=True,
        )

        for detection in ordered:
            best_id = None
            best_iou = self.match_iou
            fallback_id = None
            fallback_distance = float("inf")

            dcx, dcy = _center(detection["bbox"])

            for track_id, track in self.tracks.items():
                if track_id in matched_ids:
                    continue
                if track["class"] != detection["class"]:
                    continue

                score = iou(track["bbox"], detection["bbox"])
                if score >= best_iou:
                    best_iou = score
                    best_id = track_id
                    continue

                # When an object moves enough between sampled frames that
                # its boxes no longer overlap, use center distance.
                tx, ty = _center(track["bbox"])
                distance = ((dcx - tx) ** 2 + (dcy - ty) ** 2) ** 0.5

                tw = abs(track["bbox"][2] - track["bbox"][0])
                th = abs(track["bbox"][3] - track["bbox"][1])
                max_distance = max(tw, th, 1) * 1.5

                if distance <= max_distance and distance < fallback_distance:
                    fallback_distance = distance
                    fallback_id = track_id

            if best_id is None:
                best_id = fallback_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "bbox": detection["bbox"],
                    "class": detection["class"],
                    "missing": 0,
                }
            else:
                self.tracks[best_id]["bbox"] = detection["bbox"]
                self.tracks[best_id]["missing"] = 0

            matched_ids.add(best_id)

            item = detection.copy()
            item["track_id"] = best_id
            results.append(item)

        for track_id in list(self.tracks):
            if track_id not in matched_ids:
                self.tracks[track_id]["missing"] += 1
                if self.tracks[track_id]["missing"] > self.max_missing:
                    del self.tracks[track_id]

        return results
