class TrafficAnalyzer:
    def __init__(self):
        self.samples = []

    def add(
        self,
        frame_no,
        time_sec,
        detections,
        agreements,
        yolo_count,
        rcnn_count,
        yolo_fresh=False,
        rcnn_fresh=False,
    ):
        self.samples.append(
            {
                "frame": int(frame_no),
                "time_sec": float(time_sec),
                "vehicles": len(detections),
                "agreement": int(agreements),
                "yolo_count": int(yolo_count),
                "rcnn_count": int(rcnn_count),
                "yolo_fresh": bool(yolo_fresh),
                "rcnn_fresh": bool(rcnn_fresh),
            }
        )

    def summary(self):
        if not self.samples:
            return {
                "peak_visible_vehicles": 0,
                "traffic_level": "LOW",
                "average_visible_vehicles": 0.0,
                "model_agreement_percent": 0.0,
                "samples": [],
            }

        peak = max(x["vehicles"] for x in self.samples)
        average = sum(x["vehicles"] for x in self.samples) / len(self.samples)

        # Agreement is meaningful only when both models ran on the same frame.
        comparable = [
            x for x in self.samples
            if x["yolo_fresh"] and x["rcnn_fresh"]
        ]
        matched = sum(x["agreement"] for x in comparable)
        yolo_total = sum(x["yolo_count"] for x in comparable)
        agreement = (matched / yolo_total * 100) if yolo_total else 0.0

        level = (
            "HIGH" if peak >= 25
            else "MEDIUM" if peak >= 10
            else "LOW"
        )

        return {
            "peak_visible_vehicles": peak,
            "traffic_level": level,
            "average_visible_vehicles": round(average, 2),
            "model_agreement_percent": round(agreement, 2),
            "samples": self.samples,
        }
