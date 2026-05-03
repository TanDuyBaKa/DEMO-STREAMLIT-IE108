import difflib
import math
import re
from typing import List

import cv2
import easyocr
import numpy as np
import streamlit as st


@st.cache_resource
def build_easyocr_reader(gpu: bool = True):
    try:
        return easyocr.Reader(["en"], gpu=gpu)
    except Exception:
        return easyocr.Reader(["en"], gpu=False)


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_actor_name(text: str) -> str:
    t = normalize_text(text)
    t = t.replace("\n", " ")
    t = re.sub(r"[|/\\]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" -_:;,.<>[](){}")


def title_case_if_needed(text: str) -> str:
    if not text:
        return text
    if text.isupper():
        return text.title()
    return text


def maybe_correct_name(text: str, known_actor_names: List[str], cutoff: float = 0.72) -> str:
    t = title_case_if_needed(clean_actor_name(text))
    if not known_actor_names:
        return t
    matches = difflib.get_close_matches(t, known_actor_names, n=1, cutoff=cutoff)
    return matches[0] if matches else t


def text_letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(1, len(text))


UML_NOISE_EXACT = {
    "include", "extend", "<<include>>", "<<extend>>", "<include>", "<extend>",
    "diagram", "figure", "actor", "use case", "use_case",
}
UML_NOISE_CONTAINS = [
    "include", "extend", "use case diagram", "case diagram", "figure", "association",
    "generalization", "boundary", "system",
]


def is_valid_actor_name(text: str, min_len: int = 2) -> bool:
    t = clean_actor_name(text)
    low = t.lower()
    if len(t) < min_len:
        return False
    if low in UML_NOISE_EXACT:
        return False
    if any(bad in low for bad in UML_NOISE_CONTAINS):
        return False
    if low.count("<") + low.count(">") >= 2:
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    if re.fullmatch(r"\d+", t):
        return False
    if text_letter_ratio(t) < 0.45:
        return False
    return True


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def point_in_box(point, box) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def inside_any_box(point, boxes) -> bool:
    return any(point_in_box(point, b) for b in boxes)


def distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def clip_box(box, W: int, H: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(x1 + 1, min(W, int(round(x2))))
    y2 = max(y1 + 1, min(H, int(round(y2))))
    return [x1, y1, x2, y2]


def intersection_area(box_a, box_b) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def nearest_boundary(actor_box, boundary_boxes):
    if not boundary_boxes:
        return None
    ac = box_center(actor_box)
    best_box, best_dist = None, float("inf")
    for b in boundary_boxes:
        d = distance(ac, box_center(b))
        if d < best_dist:
            best_dist = d
            best_box = b
    return best_box


def make_actor_rois(actor_box, img_shape):
    H, W = img_shape[:2]
    x1, y1, x2, y2 = actor_box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    rois = [
        ("label_band", clip_box((x1 - 0.10*w, y1 + 0.48*h, x2 + 0.10*w, y2 + 0.08*h), W, H)),
        ("below_band", clip_box((x1 - 0.20*w, y1 + 0.72*h, x2 + 0.20*w, y2 + 0.45*h), W, H)),
        ("lower_half", clip_box((x1 - 0.08*w, y1 + 0.38*h, x2 + 0.08*w, y2 + 0.08*h), W, H)),
        ("full_box", clip_box((x1, y1, x2, y2), W, H)),
    ]
    unique, seen = [], set()
    for name, box in rois:
        key = tuple(box)
        if key not in seen:
            seen.add(key)
            unique.append((name, box))
    return unique


def preprocess_variants(crop_bgr):
    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        return {}
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    scale = max(1.0, 280.0 / min(h, w))
    scale = min(scale, 3.5)
    if scale > 1.01:
        rgb = cv2.resize(crop_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        rgb = crop_rgb.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = 255 - otsu
    return {"rgb": rgb, "gray": gray, "sharp": sharp, "otsu": otsu, "inv": inv}


def horizontal_overlap_ratio(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    wa = max(1, ax2 - ax1)
    wb = max(1, bx2 - bx1)
    return inter / min(wa, wb)


def merge_local_ocr_items(items):
    if not items:
        return []
    items = sorted(items, key=lambda z: (z["box"][1], z["box"][0]))
    merged, used = [], set()
    for i in range(len(items)):
        if i in used:
            continue
        current = items[i].copy()
        used.add(i)
        changed = True
        while changed:
            changed = False
            _, cy1, _, cy2 = current["box"]
            ch = max(1, cy2 - cy1)
            for j in range(len(items)):
                if j in used:
                    continue
                b = items[j]
                bx1, by1, bx2, by2 = b["box"]
                bh = max(1, by2 - by1)
                x_overlap = horizontal_overlap_ratio(current["box"], b["box"])
                vertical_gap = by1 - current["box"][3]
                if x_overlap >= 0.30 and 0 <= vertical_gap <= max(20, int(1.0 * max(ch, bh))):
                    nx1 = min(current["box"][0], bx1)
                    ny1 = min(current["box"][1], by1)
                    nx2 = max(current["box"][2], bx2)
                    ny2 = max(current["box"][3], by2)
                    current = {
                        "text": clean_actor_name(current["text"] + " " + b["text"]),
                        "conf": max(current["conf"], b["conf"]),
                        "box": [nx1, ny1, nx2, ny2],
                        "center": box_center([nx1, ny1, nx2, ny2]),
                        "roi_name": current["roi_name"],
                        "variant_name": current["variant_name"],
                    }
                    used.add(j)
                    changed = True
        merged.append(current)
    return merged


def ocr_candidates_from_roi(img_bgr, roi_box, roi_name, reader, known_actor_names):
    x1, y1, x2, y2 = roi_box
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    variants = preprocess_variants(crop)
    crop_h, crop_w = crop.shape[:2]
    candidates = []
    for variant_name, var_img in variants.items():
        vh, vw = var_img.shape[:2]
        sx = crop_w / max(1, vw)
        sy = crop_h / max(1, vh)
        raw = reader.readtext(var_img, detail=1, paragraph=False, decoder="greedy")
        local_items = []
        for item in raw:
            poly, text, conf = None, None, 0.0
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    poly, text, conf = item
                elif len(item) == 2:
                    poly, text = item
                    conf = 0.0
                else:
                    continue
            else:
                continue
            if poly is None:
                continue
            text = maybe_correct_name(text, known_actor_names)
            if conf is None:
                conf = 0.0
            if conf < 0.16 or not is_valid_actor_name(text):
                continue
            try:
                xs = [int(p[0]) for p in poly]
                ys = [int(p[1]) for p in poly]
            except Exception:
                continue
            lx1, ly1, lx2, ly2 = min(xs), min(ys), max(xs), max(ys)
            gx1, gy1 = int(x1 + lx1 * sx), int(y1 + ly1 * sy)
            gx2, gy2 = int(x1 + lx2 * sx), int(y1 + ly2 * sy)
            gbox = [gx1, gy1, gx2, gy2]
            local_items.append({
                "text": text,
                "conf": float(conf),
                "box": gbox,
                "center": box_center(gbox),
                "roi_name": roi_name,
                "variant_name": variant_name,
            })
        candidates.extend(merge_local_ocr_items(local_items))
    return candidates


def dedupe_candidates(candidates):
    if not candidates:
        return []
    kept = []
    for c in candidates:
        c_text = c["text"].lower()
        c_box = c["box"]
        c_area = max(1, (c_box[2] - c_box[0]) * (c_box[3] - c_box[1]))
        should_skip, to_remove = False, []
        for i, k in enumerate(kept):
            k_text = k["text"].lower()
            k_box = k["box"]
            inter = intersection_area(c_box, k_box)
            k_area = max(1, (k_box[2] - k_box[0]) * (k_box[3] - k_box[1]))
            similar = (inter / c_area > 0.35) or (inter / k_area > 0.35)
            if not similar:
                continue
            if c_text in k_text or k_text in c_text:
                if len(c_text) > len(k_text):
                    to_remove.append(i)
                else:
                    should_skip = True
            elif c_text == k_text:
                if c["conf"] > k["conf"]:
                    to_remove.append(i)
                else:
                    should_skip = True
        if not should_skip:
            for idx in reversed(to_remove):
                kept.pop(idx)
            kept.append(c)
    return kept


ROI_PRIORITY = {"label_band": 0.0, "below_band": 0.20, "lower_half": 0.45, "full_box": 1.05}


def score_local_candidate(candidate, actor_box, use_case_boxes, boundary_box):
    actor_c = box_center(actor_box)
    actor_w, actor_h = box_size(actor_box)
    text = candidate["text"]
    text_box = candidate["box"]
    text_c = candidate["center"]
    roi_name = candidate["roi_name"]
    conf = candidate["conf"]
    score = ROI_PRIORITY.get(roi_name, 1.0)
    score -= min(conf, 1.0) * 1.6
    score += abs(text_c[0] - actor_c[0]) / max(1, actor_w) * 0.30
    rel_y = text_c[1] - actor_c[1]
    if rel_y < -0.1 * actor_h:
        score += 1.7
    elif 0.0 <= rel_y <= 1.6 * actor_h:
        score -= 1.0
    elif rel_y > 2.5 * actor_h:
        score += 1.1
    if inside_any_box(text_c, use_case_boxes):
        score += 100.0
    for ub in use_case_boxes:
        inter = intersection_area(text_box, ub)
        area = max(1, (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]))
        if inter / area > 0.25:
            score += 100.0
            break
    if boundary_box is not None:
        actor_inside = point_in_box(actor_c, boundary_box)
        text_inside = point_in_box(text_c, boundary_box)
        if (not actor_inside) and text_inside:
            score += 3.0
    n_words = len(text.split())
    if 1 <= n_words <= 4:
        score -= 0.7
    else:
        score += 0.45 * abs(n_words - 3)
    score += max(0.0, 0.65 - text_letter_ratio(text)) * 3.2
    if 2 <= n_words <= 4:
        score -= 0.35
    return score


def extract_actor_names(img_bgr, actors, use_cases, boundaries, reader, known_actor_names):
    use_case_boxes = [u["box"] for u in use_cases]
    boundary_boxes = [b["box"] for b in boundaries]
    results = []
    for idx, actor in enumerate(actors, start=1):
        actor_box = actor["box"]
        boundary_box = nearest_boundary(actor_box, boundary_boxes)
        all_candidates = []
        for roi_name, roi_box in make_actor_rois(actor_box, img_bgr.shape):
            all_candidates.extend(ocr_candidates_from_roi(img_bgr, roi_box, roi_name, reader, known_actor_names))
        all_candidates = dedupe_candidates(all_candidates)
        if not all_candidates:
            results.append({
                "actor_index": idx,
                "name": "Unknown",
                "actor_conf": actor["confidence"],
                "ocr_conf": None,
                "roi_name": None,
                "box": actor_box,
            })
            continue
        for c in all_candidates:
            c["score"] = score_local_candidate(c, actor_box, use_case_boxes, boundary_box)
        best = sorted(all_candidates, key=lambda z: z["score"])[0]
        results.append({
            "actor_index": idx,
            "name": best["text"],
            "actor_conf": actor["confidence"],
            "ocr_conf": best["conf"],
            "roi_name": best["roi_name"],
            "box": actor_box,
            "candidate_box": best["box"],
            "score": best["score"],
        })
    return results
