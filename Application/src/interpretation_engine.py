def interpret(summary, counts):
    total = sum(counts.values())
    dominant = max(counts, key=counts.get) if total else "none"

    return (
        f"Traffic level is {summary['traffic_level']}. "
        f"The system counted {total} vehicles crossing the counting line, "
        f"with {dominant} as the dominant vehicle type. "
        f"The average number of visible vehicles per analyzed frame was "
        f"{summary['average_visible_vehicles']}. "
        f"YOLO and Faster R-CNN had "
        f"{summary['model_agreement_percent']}% agreement on comparable frames. "
        "The fused output combines detections from both models. "
        "This is an operational traffic-analysis result and not a benchmark "
        "accuracy score because no ground-truth annotations are provided."
    )
