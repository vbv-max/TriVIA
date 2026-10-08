import os
import shutil
from datetime import datetime
import cv2
import pandas as pd

from .yolo_detector import YOLODetector
from .rcnn_detector import FasterRCNNDetector
from .result_fusion import fuse_detections
from .vehicle_tracker import SimpleTracker
from .vehicle_counter import VehicleCounter
from .traffic_analyzer import TrafficAnalyzer
from .interpretation_engine import interpret
from .visualization import make_graphs


class VideoProcessor:
    def __init__(self, confidence=0.5, yolo_interval=3, rcnn_interval=15):
        self.confidence = float(confidence)
        self.yolo_interval = max(1, int(yolo_interval))
        self.rcnn_interval = max(1, int(rcnn_interval))

        self.yolo = YOLODetector(self.confidence)
        self.rcnn = FasterRCNNDetector(self.confidence)

    def run(self, input_path, output_root, progress_callback=None):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open input video: {input_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("The uploaded video has invalid dimensions.")

        os.makedirs(output_root, exist_ok=True)

        # NEVER delete a previous output directory on Windows. A media player,
        # antivirus scanner, Streamlit, or Explorer can still have an MP4 open,
        # which makes shutil.rmtree() fail with WinError 32.  Give every analysis
        # its own run directory instead. This also makes results reproducible and
        # prevents files from two analyses being mixed together.
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_root = os.path.join(output_root, run_id)
        paths = {
            key: os.path.join(run_root, key)
            for key in ["yolo", "rcnn", "combined", "results", "graphs"]
        }
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = {}

        for key in ["yolo", "rcnn", "combined"]:
            output_file = os.path.join(paths[key], f"{key}_result.mp4")
            writer = cv2.VideoWriter(
                output_file,
                fourcc,
                fps,
                (width, height),
            )
            if not writer.isOpened():
                cap.release()
                for old_writer in writers.values():
                    old_writer.release()
                raise RuntimeError(
                    f"Could not create output video: {output_file}"
                )
            writers[key] = writer

        tracker = SimpleTracker()
        counter = VehicleCounter()
        analyzer = TrafficAnalyzer()
        rows = []
        yolo_rows = []
        rcnn_rows = []
        comparison_rows = []

        last_yolo = []
        last_rcnn = []
        last_combined = []
        last_agreements = 0

        frame_no = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                yolo_fresh = frame_no % self.yolo_interval == 0
                rcnn_fresh = frame_no % self.rcnn_interval == 0

                if yolo_fresh:
                    last_yolo = self.yolo.detect(frame)

                if rcnn_fresh:
                    last_rcnn = self.rcnn.detect(frame)

                # Fusion is updated only when both models run on the same
                # frame. Between fusion frames we keep the last fused result
                # for stable tracking and fast rendering.
                if yolo_fresh and rcnn_fresh:
                    last_combined, agreements = fuse_detections(
                        last_yolo,
                        last_rcnn,
                    )
                    last_combined = [dict(d) for d in last_combined]
                    last_agreements = agreements
                else:
                    agreements = 0

                combined = last_combined
                current_agreements = last_agreements if (yolo_fresh and rcnn_fresh) else 0

                tracked = tracker.update(combined)
                counter.update(tracked, height)

                analyzer.add(
                    frame_no,
                    frame_no / fps,
                    tracked,
                    current_agreements,
                    len(last_yolo),
                    len(last_rcnn),
                    yolo_fresh=yolo_fresh,
                    rcnn_fresh=rcnn_fresh,
                )

                for d in tracked:
                    x1, y1, x2, y2 = d["bbox"]
                    rows.append(
                        {
                            "frame": frame_no,
                            "time_sec": round(frame_no / fps, 3),
                            "track_id": d.get("track_id"),
                            "class": d["class"],
                            "confidence": round(float(d["confidence"]), 4),
                            "agreement": bool(d.get("agreement", False)),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )

                for d in last_yolo:
                    yolo_rows.append({
                        "frame": frame_no, "time_sec": round(frame_no / fps, 3),
                        "class": d["class"], "confidence": round(float(d["confidence"]), 4),
                        "x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3],
                    })
                for d in last_rcnn:
                    rcnn_rows.append({
                        "frame": frame_no, "time_sec": round(frame_no / fps, 3),
                        "class": d["class"], "confidence": round(float(d["confidence"]), 4),
                        "x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3],
                    })

                if yolo_fresh and rcnn_fresh:
                    comparison_rows.append({
                        "frame": frame_no, "time_sec": round(frame_no / fps, 3),
                        "yolo_detections": len(last_yolo), "rcnn_detections": len(last_rcnn),
                        "matched_detections": agreements,
                    })

                yframe = self._draw(frame, last_yolo, "YOLO")
                rframe = self._draw(frame, last_rcnn, "Faster R-CNN")
                cframe = self._draw(
                    frame,
                    tracked,
                    "TRIVA FUSED",
                    show_count_line=True,
                )

                writers["yolo"].write(yframe)
                writers["rcnn"].write(rframe)
                writers["combined"].write(cframe)

                frame_no += 1

                if progress_callback and total_frames > 0:
                    progress_callback(min(frame_no / total_frames, 1.0))

        finally:
            cap.release()
            for writer in writers.values():
                writer.release()

        if frame_no == 0:
            raise RuntimeError("No readable frames were found in the video.")

        summary = analyzer.summary()
        interpretation = interpret(summary, counter.counts)

        pd.DataFrame(rows).to_csv(
            os.path.join(paths["results"], "fused_detections.csv"),
            index=False,
        )

        pd.DataFrame(
            [
                {"vehicle_type": key, "count": value}
                for key, value in counter.counts.items()
            ]
        ).to_csv(
            os.path.join(paths["results"], "traffic_summary.csv"),
            index=False,
        )

        report_path = os.path.join(
            paths["results"],
            "traffic_analysis_report.txt",
        )
        with open(report_path, "w", encoding="utf-8") as report:
            report.write(interpretation)

        pd.DataFrame(yolo_rows).to_csv(os.path.join(paths["results"], "yolo_detections.csv"), index=False)
        pd.DataFrame(rcnn_rows).to_csv(os.path.join(paths["results"], "rcnn_detections.csv"), index=False)
        pd.DataFrame(comparison_rows).to_csv(
            os.path.join(paths["results"], "model_comparison_frames.csv"),
            index=False,
        )

        comparable_frames = len(comparison_rows)
        yolo_detections = sum(r["yolo_detections"] for r in comparison_rows)
        rcnn_detections = sum(r["rcnn_detections"] for r in comparison_rows)
        matched_detections = sum(r["matched_detections"] for r in comparison_rows)
        comparison = {
            "comparable_frames": comparable_frames,
            "yolo_detections": yolo_detections,
            "rcnn_detections": rcnn_detections,
            "matched_detections": matched_detections,
            "agreement_percent": round((matched_detections / max(yolo_detections, 1)) * 100, 2),
            "class_comparison": [
                {
                    "vehicle_type": cls,
                    "YOLO": sum(1 for r in yolo_rows if r["class"] == cls),
                    "Faster R-CNN": sum(1 for r in rcnn_rows if r["class"] == cls),
                    "Combined counted": counter.counts.get(cls, 0),
                }
                for cls in ["bicycle", "car", "motorcycle", "bus", "truck"]
            ],
        }

        pd.DataFrame(comparison["class_comparison"]).to_csv(
            os.path.join(paths["results"], "model_comparison.csv"),
            index=False,
        )

        findings = [
            f"Traffic level is {summary['traffic_level']} with a peak of {summary['peak_visible_vehicles']} visible vehicles.",
            f"{sum(counter.counts.values())} vehicle crossings were counted across the five supported vehicle classes.",
            f"The average number of visible vehicles per processed frame was {summary['average_visible_vehicles']}.",
            f"YOLO and Faster R-CNN matched {matched_detections} detections across {comparable_frames} comparable frames.",
        ]
        if counter.counts:
            dominant = max(counter.counts, key=counter.counts.get)
            findings.append(f"The dominant counted vehicle type was {dominant} ({counter.counts[dominant]} crossings).")

        make_graphs(counter.counts, summary, paths["graphs"])

        return {
            "summary": summary,
            "counts": counter.counts,
            "interpretation": interpretation,
            "paths": paths,
            "fps": round(frame_no / fps, 3),
            "frames_processed": frame_no,
            "comparison": comparison,
            "findings": findings,
        }

    @staticmethod
    def _draw(frame, detections, title, show_count_line=False):
        output = frame.copy()
        height, width = output.shape[:2]

        cv2.putText(
            output,
            title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        if show_count_line:
            line_y = int(height * 0.60)
            cv2.line(
                output,
                (0, line_y),
                (width, line_y),
                (255, 0, 255),
                2,
            )
            cv2.putText(
                output,
                "COUNTING LINE",
                (20, max(55, line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            color = (
                (0, 255, 0)
                if d.get("agreement")
                else (0, 255, 255)
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = (
                f"{d['class']} "
                f"{float(d['confidence']):.2f}"
            )

            if d.get("track_id") is not None:
                label += f" ID:{d['track_id']}"

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return output
