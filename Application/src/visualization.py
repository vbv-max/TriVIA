import os
import matplotlib.pyplot as plt


def make_graphs(counts, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title("Vehicle Count")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vehicle_count.png"), dpi=150)
    plt.close()

    non_zero = [(k, v) for k, v in counts.items() if v > 0]
    plt.figure(figsize=(6, 6))
    if non_zero:
        plt.pie(
            [v for _, v in non_zero],
            labels=[k for k, _ in non_zero],
            autopct="%1.1f%%",
        )
        plt.title("Vehicle Composition")
    else:
        plt.text(0.5, 0.5, "No vehicles counted", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vehicle_composition.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["YOLO + R-CNN Agreement"],
        [summary["model_agreement_percent"]],
    )
    plt.ylim(0, 100)
    plt.ylabel("Percent")
    plt.title("Model Agreement")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_agreement.png"), dpi=150)
    plt.close()

    samples = summary.get("samples", [])
    if samples:
        x = [s["time_sec"] for s in samples]
        y = [s["vehicles"] for s in samples]

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, marker="o", markersize=2)
        plt.title("Traffic Flow")
        plt.xlabel("Time (s)")
        plt.ylabel("Visible Vehicles")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "traffic_flow.png"), dpi=150)
        plt.close()
