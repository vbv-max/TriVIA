from ultralytics import YOLO

YOLO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YOLODetector:
    def __init__(self, confidence=0.5, model_path="yolo11n.pt", imgsz=512):
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.model = YOLO(model_path)

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []

        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections = []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            class_id = int(cls)
            if class_id not in YOLO_VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": YOLO_VEHICLE_CLASSES[class_id],
                    "confidence": float(conf),
                    "model": "YOLO",
                }
            )

        return detections
