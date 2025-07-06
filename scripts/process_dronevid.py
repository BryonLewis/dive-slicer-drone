import os
import re
import cv2
import json
import click
import numpy as np
import pytesseract
import easyocr
import logging
from tqdm import tqdm
from datetime import timedelta
from collections import deque

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
)

# OCR configuration
pytesseract.pytesseract.tesseract_cmd = 'tesseract'
tess_config = '--psm 7'
tess_numeric_config = '--psm 7 -c tessedit_char_whitelist=0123456789.'

# EasyOCR setup
reader = easyocr.Reader(['en'], gpu=True)

# Template setup
BATTERY_TEMPLATE_PATHS = ["battery.png"]
THROTTLE_TEMPLATE_PATHS = ["throttle.png"]
MATCH_THRESHOLD = 0.7
seen_status_values = set()

def load_templates(paths):
    templates = []
    for path in paths:
        if os.path.exists(path):
            templates.append(cv2.imread(path, 0))
            logging.info(f"Loaded template: {path}")
        else:
            logging.warning(f"Template not found: {path}")
    return templates

def locate_template_in_frame(gray_frame, template):
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    h, w = template.shape
    return max_loc, max_val, w, h

def ocr_from_box_tesseract(frame, box, numeric_only=False):
    if box is None:
        return "", -1
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)
    config = tess_numeric_config if numeric_only else tess_config
    try:
        data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(thresh, config=config).strip()
        if numeric_only:
            text = re.sub(r'[^0-9.]', '', text)
        conf = max([int(c) for c in data['conf'] if c != '-1'] or [-1])
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return "", -1
    return text, conf

def is_valid_voltage(text):
    return bool(re.fullmatch(r'\d+\.\d{1,2}', text))

def read_center_status_easyocr(frame):
    x, y, w, h = 400, 320, 500, 200
    roi = frame[y:y+h, x:x+w]
    results = reader.readtext(roi, detail=0)
    possible_results = []

    for result in results:
        upper = result.upper()
        seen_status_values.add(upper)

        if "CRASH" in upper:
            possible_results.append(("FLIP OVER CRASHED", (x, y, w, h)))
        elif "CURRENT" in upper or "GURRENT" in upper:
            possible_results.append(("FLIGHT DONE", (x, y, w, h)))
        elif "BATTERY" in upper:
            possible_results.append(("LOW BATTERY", (x, y, w, h)))
        elif "ARMED" in upper:
            if upper.startswith("D"):
                possible_results.append(("DISARMED", (x, y, w, h)))
            elif upper.startswith("A"):
                possible_results.append(("ARMED", (x, y, w, h)))

    for priority in ["FLIP OVER CRASHED", "LOW BATTERY", "ARMED", "FLIGHT DONE"]:
        for item in possible_results:
            if item == priority:
                return item
    return possible_results[0] if possible_results else (None, (0, 0, 0, 0))

def create_full_track(width, height):
    return {
        "id": 0,
        "begin": 0,
        "end": 0,
        "confidencePairs": [["unknown", 1.0]],
        "meta": {"time": True},
        "attributes": {},
        "features": [],
    }

@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("output_json", default="output_frames.json", type=click.Path())
@click.option("--debug", is_flag=True, help="Enable debug mode and save annotated frames.")
@click.option("--frame_step", default=10, show_default=True, help="Process every Nth frame.")
def main(video_path, output_json, debug, frame_step):
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    battery_templates = load_templates(BATTERY_TEMPLATE_PATHS)
    throttle_templates = load_templates(THROTTLE_TEMPLATE_PATHS)

    if debug:
        os.makedirs("debug_combined", exist_ok=True)

    battery_locations = deque(maxlen=5)
    throttle_locations = deque(maxlen=5)
    battery_box_cached = None
    throttle_box_cached = None

    track_obj = create_full_track(width, height)
    results = []

    frame_idx = 0
    maxFrame = -1
    pbar = tqdm(total=total_frames, desc="Processing frames")

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        maxFrame = max(maxFrame, frame_idx)

        if frame_idx % frame_step != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        timestamp = str(timedelta(seconds=frame_idx / fps))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Battery box detection
        voltage_box = battery_box_cached
        if not battery_box_cached:
            best_score, best_loc = -1, None
            for template in battery_templates:
                loc, score, w, h = locate_template_in_frame(gray, template)
                if score > best_score and score >= MATCH_THRESHOLD:
                    best_score = score
                    best_loc = (loc[0] + w, loc[1], round(w * 4.5), h)
            if best_loc:
                battery_locations.append(best_loc)
                if len(battery_locations) == 2 and all(loc == battery_locations[0] for loc in battery_locations):
                    battery_box_cached = battery_locations[0]
                voltage_box = best_loc
            else:
                battery_locations.append(None)

        # Throttle box detection
        throttle_box = throttle_box_cached
        if not throttle_box_cached:
            best_score, best_loc = -1, None
            for template in throttle_templates:
                loc, score, w, h = locate_template_in_frame(gray, template)
                if score > best_score and score >= MATCH_THRESHOLD:
                    best_score = score
                    best_loc = (loc[0] + (w * 2), loc[1] + round(h * 0.25), w * 2, round(h * 0.75))
            if best_loc:
                throttle_locations.append(best_loc)
                if len(throttle_locations) == 2 and all(loc == throttle_locations[0] for loc in throttle_locations):
                    throttle_box_cached = throttle_locations[0]
                throttle_box = best_loc
            else:
                throttle_locations.append(None)

        voltage_text, voltage_conf = ocr_from_box_tesseract(frame, voltage_box, numeric_only=True) if voltage_box else (None, -1)
        if voltage_text and not is_valid_voltage(voltage_text):
            voltage_text = None

        throttle_text, throttle_conf = ocr_from_box_tesseract(frame, throttle_box, numeric_only=True) if throttle_box else (None, -1)
        if not throttle_text and throttle_box is not None:
            throttle_text = "0"

        status_text, status_box = read_center_status_easyocr(frame)

        logging.info(f"[{timestamp}] Voltage: {voltage_text}, Throttle: {throttle_text}, Status: {status_text}")

        if debug:
            annotated = frame.copy()
            def draw_box(box, label, conf=None):
                if box:
                    x, y, w, h = map(int, box)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text_label = f"{label or 'N/A'} ({conf}%)" if conf is not None else label
                    cv2.putText(annotated, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            draw_box(voltage_box, voltage_text, voltage_conf)
            draw_box(throttle_box, throttle_text, throttle_conf)
            draw_box(status_box, status_text, 100)
            cv2.imwrite(os.path.join("debug_combined", f"frame_{frame_idx:05d}.jpg"), annotated)

        attributes = {}
        throttle_val = safe_float(throttle_text)
        voltage_val = safe_float(voltage_text)
        if status_text:
            attributes["status"] = status_text
        if throttle_val is not None and throttle_val <= 100:
            attributes["throttle"] = throttle_val
        if voltage_val is not None:
            attributes["voltage"] = voltage_val

        track_obj["features"].append({
            "frame": frame_idx,
            "keyframe": True,
            "bounds": [0, 0, width, height],
            "attributes": attributes,
        })

        results.append({
            "time": timestamp,
            "voltage": voltage_text,
            "throttle": throttle_text,
            "status": status_text,
        })

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Interpolation
    features = track_obj["features"]
    if features[-1]["frame"] != maxFrame:
        features.append({
            "frame": maxFrame,
            "keyframe": True,
            "bounds": [0, 0, width, height],
            "attributes": {}
        })
    track_obj["end"] = maxFrame
    interpolated = []
    for i in range(len(features) - 1):
        interpolated.append(features[i])
        f0, f1 = features[i], features[i+1]
        gap = f1["frame"] - f0["frame"]
        if gap > 1:
            for j in range(1, gap):
                alpha = j / gap
                iv = {}
                for key in ("voltage", "throttle"):
                    v0, v1 = f0["attributes"].get(key), f1["attributes"].get(key)
                    if v0 is not None and v1 is not None:
                        iv[key] = round((1 - alpha) * v0 + alpha * v1, 2)
                interpolated.append({
                    "frame": f0["frame"] + j,
                    "keyframe": True,
                    "bounds": [0, 0, width, height],
                    "attributes": iv,
                })
    interpolated.append(features[-1])

    # Remove duplicate statuses and track last seen
    last_status = (-1, None)
    for feat in interpolated:
        status = feat["attributes"].get("status")
        if status is not None and status != last_status[1]:
            last_status = (feat["frame"], status)
        else:
            feat["attributes"].pop("status", None)

    # Ensure last feature has the last known status
    if last_status:
        interpolated[-1]["attributes"]["status"] = last_status[1]

    track_obj["features"] = interpolated

    with open(output_json, 'w') as f:
        json.dump({
            "tracks": {"0": track_obj},
            "groups": {},
            "version": 2,
        }, f, indent=2)

    logging.info(f"Processing complete. Output: {output_json}")

    # Compute stats
    non_zero_throttles = [feat["attributes"]["throttle"]
                          for feat in interpolated
                          if "throttle" in feat["attributes"] and feat["attributes"]["throttle"] > 0]
    avg_throttle = round(sum(non_zero_throttles) / len(non_zero_throttles), 2) if non_zero_throttles else 0.0

    crash_count = 0
    i = 0
    crash_window = int(10 * fps)

    while i < len(interpolated):
        feat = interpolated[i]
        status = feat["attributes"].get("status")
        if status == "FLIP OVER CRASHED":
            j = i + 1
            armed_count = 0
            while j < len(interpolated) and (interpolated[j]["frame"] - feat["frame"]) <= crash_window:
                next_status = interpolated[j]["attributes"].get("status")
                if next_status == "ARMED" or next_status == "LOW BATTERY":
                    armed_count += 1
                elif next_status and next_status != "ARMED" and next_status != "LOW BATTERY":
                    break  # Status changed to something else
                j += 1
            if armed_count == (j - i):  # All statuses were ARMED within the window
                crash_count += 1
                i = j  # Skip ahead to avoid recounting
            else:
                i += 1
        else:
            i += 1

    stats = {
        "average_throttle_above_zero": avg_throttle,
        "crash_count": crash_count,
    }

    stats_out = os.path.splitext(output_json)[0] + "_stats.json"
    with open(stats_out, 'w') as f:
        json.dump(stats, f, indent=2)

    logging.info(f"Stats saved to: {stats_out}")

if __name__ == "__main__":
    main()
