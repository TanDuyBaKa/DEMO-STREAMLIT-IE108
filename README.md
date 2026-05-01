# UML Actor Detection Demo

Demo nhỏ cho đồ án: phát hiện và trích xuất Actor từ sơ đồ UML Use Case bằng YOLOv8 + OCR.

## Chức năng

- Upload ảnh UML Use Case.
- YOLOv8 phát hiện Actor, Use_Case, Use_Case_Boundary.
- EasyOCR đọc tên actor.
- Hiển thị ảnh kết quả.
- Hiển thị bảng actor.
- Tải kết quả CSV/JSON.

## Cấu trúc thư mục

```text
uml_actor_streamlit_demo/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── best.pt
├── sample_images/
├── outputs/
└── src/
    ├── ocr_utils.py
    └── visualization.py
```

## Chuẩn bị model

Copy file `best.pt` sau khi train YOLO vào:

```text
models/best.pt
```

Nếu chưa copy model, demo vẫn cho phép upload `best.pt` ở sidebar.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy demo

```bash
streamlit run app.py
```

Sau đó mở link localhost do Streamlit hiển thị.

## Upload lên GitHub

```bash
git init
git add .
git commit -m "Initial UML actor detection demo"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Lưu ý: nếu `best.pt` quá lớn, không nên commit trực tiếp lên GitHub. Có thể upload model lên Google Drive/Release và ghi hướng dẫn tải trong README.
