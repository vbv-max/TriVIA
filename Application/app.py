import io
import os
import zipfile
import streamlit as st

from src.video_processor import VideoProcessor

PROJECT_NAME = "TriVIA: Traffic Vision & Intelligent Analytics"
OUTPUT_ROOT = "outputs"

st.set_page_config(page_title=PROJECT_NAME, page_icon="🚦", layout="wide")

st.title("🚦 TriVIA: Traffic Vision & Intelligent Analytics")
st.caption("YOLO + Faster R-CNN + intelligent fusion, tracking, counting and traffic interpretation")

os.makedirs("data/input", exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

if "trivia_result" not in st.session_state:
    st.session_state.trivia_result = None
if "trivia_video_name" not in st.session_state:
    st.session_state.trivia_video_name = None

video = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])

col1, col2, col3 = st.columns(3)
with col1:
    yolo_interval = st.number_input("YOLO frame interval", 1, 20, 3, 1)
with col2:
    rcnn_interval = st.number_input("Faster R-CNN frame interval", 1, 30, 15, 1)
with col3:
    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)

if st.button("🚦 Analyze Traffic", type="primary", use_container_width=True):
    if video is None:
        st.error("Please upload a traffic video.")
    else:
        input_path = "data/input/input_video.mp4"
        with open(input_path, "wb") as file:
            file.write(video.getbuffer())

        progress = st.progress(0)
        status = st.empty()
        status.info("Loading YOLO + Faster R-CNN and analyzing video. First run may download the Faster R-CNN weights.")

        try:
            processor = VideoProcessor(
                confidence=confidence,
                yolo_interval=yolo_interval,
                rcnn_interval=rcnn_interval,
            )
            result = processor.run(
                input_path,
                OUTPUT_ROOT,
                progress_callback=progress.progress,
            )
            result["video_name"] = video.name
            result["settings"] = {
                "confidence": confidence,
                "yolo_interval": yolo_interval,
                "rcnn_interval": rcnn_interval,
            }
            st.session_state.trivia_result = result
            st.session_state.trivia_video_name = video.name
            progress.progress(1.0)
            status.success("Analysis completed successfully. Results are ready in all six tabs.")
        except Exception as error:
            st.session_state.trivia_result = None
            status.error("Analysis failed.")
            st.exception(error)

result = st.session_state.trivia_result

if result:
    summary = result["summary"]
    counts = result["counts"]
    paths = result["paths"]
    comparison = result.get("comparison", {})

    st.divider()
    st.subheader("Analysis Summary")
    metrics = [
        ("Vehicles Counted", sum(counts.values())),
        ("Cars", counts.get("car", 0)),
        ("Motorcycles", counts.get("motorcycle", 0)),
        ("Buses", counts.get("bus", 0)),
        ("Trucks", counts.get("truck", 0)),
        ("Bicycles", counts.get("bicycle", 0)),
        ("Traffic Level", summary["traffic_level"]),
        ("Model Agreement", f"{summary['model_agreement_percent']}%"),
    ]
    cols = st.columns(8)
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    tabs = st.tabs([
        "⭐ Combined Result",
        "🤖 YOLO",
        "🧠 Faster R-CNN",
        "📈 Visual Analysis",
        "🔍 Model Comparison",
        "💡 Interpretation",
    ])

    def read_bytes(path):
        with open(path, "rb") as file:
            return file.read()

    def show_video_result(tab, path, title):
        # Video playback is intentionally disabled. The generated MP4 remains
        # available as a download so the dashboard never embeds a video player.
        with tab:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                st.subheader(title)
                st.success("Result video generated successfully.")
                st.caption("Video preview is disabled. Download the MP4 below to view the result.")
                st.download_button(
                    f"⬇️ Download {os.path.basename(path)}",
                    data=read_bytes(path),
                    file_name=os.path.basename(path),
                    mime="video/mp4",
                    key=f"video_dl_{path}",
                    use_container_width=True,
                )
            else:
                st.error(f"{title} was not generated. Expected file: {path}")

    show_video_result(tabs[0], os.path.join(paths["combined"], "combined_result.mp4"), "Combined YOLO + Faster R-CNN Result")
    show_video_result(tabs[1], os.path.join(paths["yolo"], "yolo_result.mp4"), "YOLO Detection Result")
    show_video_result(tabs[2], os.path.join(paths["rcnn"], "rcnn_result.mp4"), "Faster R-CNN Detection Result")

    with tabs[3]:
        st.subheader("Visual Traffic Analysis")
        graph_names = [
            ("vehicle_count.png", "Vehicle Count"),
            ("vehicle_composition.png", "Vehicle Composition"),
            ("model_agreement.png", "Model Agreement"),
            ("traffic_flow.png", "Traffic Flow"),
        ]
        for filename, title in graph_names:
            path = os.path.join(paths["graphs"], filename)
            if os.path.isfile(path):
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
                st.download_button(
                    f"⬇️ Download {filename}",
                    data=read_bytes(path),
                    file_name=filename,
                    mime="image/png",
                    key=f"graph_dl_{filename}",
                )
            else:
                st.warning(f"{title} is unavailable.")

    with tabs[4]:
        st.subheader("YOLO vs Faster R-CNN")
        st.caption("Agreement is an IoU-based comparison on frames where both models actually ran. It is not accuracy, precision, recall, F1 or mAP.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Comparable Frames", comparison.get("comparable_frames", 0))
        c2.metric("YOLO Detections", comparison.get("yolo_detections", 0))
        c3.metric("Faster R-CNN Detections", comparison.get("rcnn_detections", 0))
        c4.metric("Matched Detections", comparison.get("matched_detections", 0))
        st.metric("Agreement", f"{comparison.get('agreement_percent', summary['model_agreement_percent'])}%")

        class_rows = comparison.get("class_comparison", [])
        if class_rows:
            st.markdown("### Detection comparison by vehicle type")
            st.dataframe(class_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No same-frame detections were available for comparison.")

        csv_path = os.path.join(paths["results"], "model_comparison.csv")
        if os.path.isfile(csv_path):
            st.download_button("⬇️ Download Model Comparison CSV", open(csv_path, "rb").read(), "model_comparison.csv", "text/csv", key="comparison_csv", use_container_width=True)

    with tabs[5]:
        st.subheader("Traffic Interpretation")
        st.info(result["interpretation"])
        st.markdown("### Key findings")
        findings = result.get("findings", [])
        if findings:
            for finding in findings:
                st.write(f"• {finding}")
        else:
            st.write("• No additional findings were generated.")

        report_path = os.path.join(paths["results"], "traffic_analysis_report.txt")
        if os.path.isfile(report_path):
            st.download_button("⬇️ Download Interpretation Report", open(report_path, "rb").read(), "traffic_analysis_report.txt", "text/plain", key="interpretation_report", use_container_width=True)

    st.divider()
    st.subheader("Downloads")
    st.caption("All downloads below belong to the current analysis and remain available when you switch tabs.")

    # Only expose files from the current analysis run. Previous runs are kept
    # on disk so Windows file locks can never break a new analysis.
    current_root = os.path.commonpath(list(paths.values()))
    files = []
    for root, _, names in os.walk(current_root):
        for name in names:
            full = os.path.join(root, name)
            if os.path.isfile(full):
                files.append(full)

    if files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for full in files:
                zf.write(full, os.path.relpath(full, OUTPUT_ROOT))
        zip_buffer.seek(0)
        st.download_button(
            "📦 Download ALL TriVIA Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="TriVIA_traffic_analysis_results.zip",
            mime="application/zip",
            key="all_results_zip",
            use_container_width=True,
        )

        download_cols = st.columns(3)
        for index, full in enumerate(files):
            with download_cols[index % 3]:
                rel = os.path.relpath(full, OUTPUT_ROOT)
                mime = "video/mp4" if full.endswith(".mp4") else "text/csv" if full.endswith(".csv") else "image/png" if full.endswith(".png") else "text/plain" if full.endswith(".txt") else "application/octet-stream"
                st.download_button(
                    f"⬇️ {rel}",
                    data=read_bytes(full),
                    file_name=os.path.basename(full),
                    mime=mime,
                    key=f"all_file_{index}",
                )
    else:
        st.warning("No downloadable result files were generated.")
else:
    st.info("Upload a traffic video and click **Analyze Traffic** to generate all six result tabs.")
