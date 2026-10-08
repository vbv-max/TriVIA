import cv2
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

RCNN_VEHICLE_CLASSES = {
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


class FasterRCNNDetector:
    """CPU-friendly Faster R-CNN detector.

    The detector keeps the official COCO pretrained model but limits the
    inference image size and number of returned detections for practical
    video processing on CPU.
    """

    def __init__(self, confidence=0.5, max_side=960):
        self.confidence = float(confidence)
        self.max_side = int(max_side)
        self.device = torch.device("cpu")

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        self.transform = weights.transforms()

    @torch.inference_mode()
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []

        original_h, original_w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(original_h, original_w))

        if scale < 1.0:
            new_w = max(1, int(original_w * scale))
            new_h = max(1, int(original_h * scale))
            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        tensor = self.transform(tensor).to(self.device)

        output = self.model([tensor])[0]
        detections = []

        for box, label, score in zip(
            output["boxes"], output["labels"], output["scores"]
        ):
            score = float(score)
            label = int(label)

            if score < self.confidence or label not in RCNN_VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.tolist()

            if scale < 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale

            x1 = max(0, min(original_w - 1, int(x1)))
            y1 = max(0, min(original_h - 1, int(y1)))
            x2 = max(0, min(original_w - 1, int(x2)))
            y2 = max(0, min(original_h - 1, int(y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": RCNN_VEHICLE_CLASSES[label],
                    "confidence": score,
                    "model": "Faster R-CNN",
                }
            )

        return detections
