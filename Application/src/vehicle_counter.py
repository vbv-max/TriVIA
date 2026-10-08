class VehicleCounter:
    def __init__(self, line_ratio=0.60):
        self.line_ratio = float(line_ratio)
        self.counted = set()
        self.previous_centers = {}
        self.counts = {
            "bicycle": 0,
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

    def update(self, detections, height):
        line = int(height * self.line_ratio)
        active_ids = set()

        for d in detections:
            track_id = d.get("track_id")
            if track_id is None:
                continue

            x1, y1, x2, y2 = d["bbox"]
            center_y = int((y1 + y2) / 2)
            active_ids.add(track_id)

            previous_y = self.previous_centers.get(track_id)
            crossed = (
                previous_y is not None
                and (
                    (previous_y < line <= center_y)
                    or (previous_y > line >= center_y)
                )
            )

            if crossed and track_id not in self.counted:
                self.counted.add(track_id)
                vehicle_class = d["class"]
                self.counts[vehicle_class] = self.counts.get(vehicle_class, 0) + 1

            self.previous_centers[track_id] = center_y

        # Prevent unbounded growth for old tracks.
        self.previous_centers = {
            tid: y for tid, y in self.previous_centers.items()
            if tid in active_ids
        }

        return self.counts.copy()
