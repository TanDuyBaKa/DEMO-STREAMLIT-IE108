import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.ocr_utils import build_easyocr_reader, extract_actor_names
from src.visualization import draw_results

APP_TITLE = "UML Actor Detection Demo"
DEFAULT_MODEL_PATH = Path("models/best.pt")

st.set_page_config(page_title=APP_TITLE, page_icon="📌", layout="wide")


@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detections_from_yolo_result(result, names):
    detections = []
    if result.boxes is None:
        return detections

    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())
        conf = float(result.boxes.conf[i].item())
        x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
        cls_name = str(names[cls_id])
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "class_name_lower": cls_name.lower(),
            "confidence": conf,
            "box": [x1, y1, x2, y2],
        })
    return detections


def split_detections(detections):
    actors, use_cases, boundaries, others = [], [], [], []
    for det in detections:
        name = det["class_name_lower"]
        if name == "actor":
            actors.append(det)
        elif name in ["use_case", "use case", "usecase"]:
            use_cases.append(det)
        elif name in ["use_case_boundary", "boundary", "system_boundary", "system boundary"]:
            boundaries.append(det)
        else:
            others.append(det)
    return actors, use_cases, boundaries, others


def build_download_json(actor_results, detections):
    payload = {"actors": actor_results, "detections": detections}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main():
    st.title("📌 UML Use Case Actor Detection Demo")
    st.caption("Demo nhỏ: YOLOv8 phát hiện Actor / Use Case / Boundary, sau đó OCR tên Actor.")

    with st.sidebar:
        st.header("Cấu hình")
        st.subheader("1. Model")
        model_path = DEFAULT_MODEL_PATH
        uploaded_model = st.file_uploader("Upload best.pt nếu chưa đặt trong thư mục models/", type=["pt"])
        if uploaded_model is not None:
            cache_dir = Path(".cache")
            cache_dir.mkdir(exist_ok=True)
            model_path = cache_dir / "uploaded_best.pt"
            model_path.write_bytes(uploaded_model.getbuffer())
        st.write("Model path:", str(model_path))

        st.subheader("2. Ngưỡng YOLO")
        conf_thres = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        iou_thres = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)

        st.subheader("3. OCR")
        enable_ocr = st.checkbox("Bật OCR trích xuất tên actor", value=True)
        use_gpu_ocr = st.checkbox("Dùng GPU cho OCR nếu có", value=True)
        known_names_text = st.text_area(
            "KNOWN_ACTOR_NAMES, mỗi dòng một tên",
            value="\n".join([
                "Student", "International Student", "Customer", "Bank", "Reception Staff",
                "Booking Process Clerk", "Cook", "Waiter", "Webmaster", "Site user",
                "Librarian", "Policeman", "Supervisor", "Tour Group Customer",
                "Individual Customer", "User", "Admin", "Administrator",
            ]),
            height=180,
        )
        known_actor_names = [x.strip() for x in known_names_text.splitlines() if x.strip()]

        st.subheader("4. Hiển thị")
        show_use_case = st.checkbox("Vẽ Use Case box", value=True)
        show_boundary = st.checkbox("Vẽ Boundary box", value=True)

    if not Path(model_path).exists():
        st.warning("Chưa tìm thấy model. Hãy đặt file `best.pt` vào `models/best.pt` hoặc upload model ở sidebar.")
        st.stop()

    uploaded_image = st.file_uploader("Upload ảnh UML Use Case", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_image is None:
        st.info("Hãy upload một ảnh UML Use Case để chạy demo.")
        st.stop()

    image = Image.open(uploaded_image).convert("RGB")
    img_bgr = pil_to_bgr(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(image, use_container_width=True)

    with st.spinner("Đang load YOLO model..."):
        model = load_yolo_model(str(model_path))

    with st.spinner("Đang phát hiện đối tượng bằng YOLOv8..."):
        results = model.predict(source=img_bgr, conf=conf_thres, iou=iou_thres, verbose=False)

    detections = detections_from_yolo_result(results[0], model.names)
    actors, use_cases, boundaries, _ = split_detections(detections)

    if enable_ocr:
        with st.spinner("Đang load EasyOCR..."):
            reader = build_easyocr_reader(gpu=use_gpu_ocr)
        with st.spinner("Đang OCR tên actor..."):
            actor_results = extract_actor_names(
                img_bgr=img_bgr,
                actors=actors,
                use_cases=use_cases,
                boundaries=boundaries,
                reader=reader,
                known_actor_names=known_actor_names,
            )
    else:
        actor_results = []
        for idx, actor in enumerate(actors, start=1):
            actor_results.append({
                "actor_index": idx,
                "name": "OCR disabled",
                "actor_conf": actor["confidence"],
                "ocr_conf": None,
                "roi_name": None,
                "box": actor["box"],
            })

    annotated = draw_results(
        img_bgr=img_bgr,
        detections=detections,
        actor_results=actor_results,
        show_use_case=show_use_case,
        show_boundary=show_boundary,
    )

    with col2:
        st.subheader("Kết quả")
        st.image(bgr_to_pil(annotated), use_container_width=True)

    st.subheader("Danh sách Actor")
    if actor_results:
        df = pd.DataFrame([{
            "actor_index": item.get("actor_index"),
            "name": item.get("name"),
            "actor_conf": item.get("actor_conf"),
            "ocr_conf": item.get("ocr_conf"),
            "roi_name": item.get("roi_name"),
            "box": item.get("box"),
        } for item in actor_results])
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Tải CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="actor_results.csv", mime="text/csv")
        st.download_button("⬇️ Tải JSON", data=build_download_json(actor_results, detections).encode("utf-8"), file_name="actor_results.json", mime="application/json")
    else:
        st.warning("Không phát hiện được Actor nào.")

    st.subheader("Tất cả detection")
    if detections:
        det_df = pd.DataFrame([{"class_name": d["class_name"], "confidence": d["confidence"], "box": d["box"]} for d in detections])
        st.dataframe(det_df, use_container_width=True)
    else:
        st.warning("Không phát hiện được đối tượng nào.")

    st.caption("Lưu ý: demo này phục vụ minh họa đồ án, chưa tối ưu cho triển khai production.")


if __name__ == "__main__":
    main()
