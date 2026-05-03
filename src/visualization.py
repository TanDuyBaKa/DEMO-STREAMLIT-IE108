import cv2


def color_for_class(class_name: str):
    name = class_name.lower()
    if name == "actor":
        return (0, 0, 255)
    if name in ["use_case", "use case", "usecase"]:
        return (255, 0, 255)
    if name in ["use_case_boundary", "boundary", "system_boundary", "system boundary"]:
        return (0, 180, 0)
    return (255, 180, 0)


def should_draw_detection(det, show_use_case=True, show_boundary=True):
    name = det["class_name_lower"]
    if name == "actor":
        return True
    if name in ["use_case", "use case", "usecase"]:
        return show_use_case
    if name in ["use_case_boundary", "boundary", "system_boundary", "system boundary"]:
        return show_boundary
    return True


def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, th + 8)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y + 4), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_results(img_bgr, detections, actor_results, show_use_case=True, show_boundary=True):
    vis = img_bgr.copy()

    for det in detections:
        if det["class_name_lower"] == "actor":
            continue
        if not should_draw_detection(det, show_use_case, show_boundary):
            continue
        x1, y1, x2, y2 = det["box"]
        color = color_for_class(det["class_name"])
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        draw_label(vis, label, x1, y1, color)

    for item in actor_results:
        x1, y1, x2, y2 = item["box"]
        color = (0, 0, 255)
        name = item.get("name", "Unknown")
        conf = item.get("actor_conf")
        label = f'{name} | actor {conf:.2f}' if conf is not None else str(name)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        draw_label(vis, label, x1, y1, color)
    return vis
