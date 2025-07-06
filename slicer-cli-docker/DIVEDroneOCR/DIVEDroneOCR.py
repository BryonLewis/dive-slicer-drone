import json
import logging
import os
import pprint
import time
from pathlib import Path
import tempfile

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

import girder_client
from datetime import timedelta

from ctk_cli import CLIArgumentParser  # noqa I004
# imported for side effects
from slicer_cli_web import ctk_cli_adjustment  # noqa
from glob import glob
from statistics import mean

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
reader = easyocr.Reader(['en'], gpu=False)

SCRIPT_DIR = Path(__file__).resolve().parent

# Template setup
BATTERY_TEMPLATE_PATHS = [SCRIPT_DIR / "ocr-images" / "battery.png"]
THROTTLE_TEMPLATE_PATHS = [SCRIPT_DIR / "ocr-images" / "throttle.png"]
MATCH_THRESHOLD = 0.7


def process_input_args(args, gc):
    folderId = args.DIVEDirectory.split('/')[-2]
    frame_step = args.FrameStep
    if not frame_step:
        frame_step = 10
    data = process_dive_dataset(gc, folderId, frame_step)
    process_metadata(args, gc, data['stats'])


def process_video(path: str, frame_step: int):
    logging.info(f"Processing video: {path}")
    cap = cv2.VideoCapture(path)
    # Check if the video file was opened successfully
    if not cap.isOpened():
        logging.info("Error: Could not open video file")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Print the results
    logging.info(f"Video Width: {width}")
    logging.info(f"Video Height: {height}")
    logging.info(f"Number of Frames: {total_frames}")

    battery_templates = load_templates(BATTERY_TEMPLATE_PATHS)
    throttle_templates = load_templates(THROTTLE_TEMPLATE_PATHS)

    battery_locations = deque(maxlen=5)
    throttle_locations = deque(maxlen=5)
    battery_box_cached = None
    throttle_box_cached = None
    track_obj = create_full_track()
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

    track_json = {
        "tracks": {"0": track_obj},
        "groups": {},
        "version": 2,
    }
    stats = calculate_stats(track_obj['features'])
    return {"width": width, "height": height, "frames": total_frames, "track_json": track_json, "stats": stats }


def calculate_stats(features):
    throttle_vals = [f["attributes"]["throttle"]
                    for f in features if "throttle" in f["attributes"] and f["attributes"]["throttle"] > 0]
    voltage_vals = [f["attributes"]["voltage"]
                    for f in features if "voltage" in f["attributes"]]

    avg_throttle = round(mean(throttle_vals), 2) if throttle_vals else 0.0
    avg_voltage = round(mean(voltage_vals), 2) if voltage_vals else 0.0

    crash_count = 0
    recovered_crash = 0
    i = 0
    total_frames = len(features)

    while i < total_frames:
        status = features[i]["attributes"].get("status")
        if status == "FLIP OVER CRASHED":
            crash_count += 1
            j = i + 1
            recovery_started = False
            recovery_success = False
            armed_frames = 0

            while j < total_frames:
                next_status = features[j]["attributes"].get("status")

                if next_status == "ARMED":
                    if not recovery_started:
                        recovery_started = True
                        armed_frames = 0
                    armed_frames += 1
                if armed_frames > 0 and not next_status:
                    armed_frames += 1
                elif next_status in ("DISARMED", "FLIP OVER CRASHED", "FLIGHT DONE"):
                    # If already recovered, this is unrelated
                    recovery_started = False
                    armed_frames = 0
                if armed_frames >= 150:
                    recovery_success = True
                    recovery_started = False
                    i = j
                    break

                j += 1
            # Also count as recovered if ARMING continues until the end
            if recovery_started and armed_frames >= 150:
                recovery_success = True
            elif recovery_started and j == total_frames:
                recovery_success = True

            if recovery_success:
                recovered_crash += 1

            i = j  # Skip ahead past the crash window
        else:
            i += 1

    return {
        "avg_throttle": avg_throttle,
        "avg_voltage": avg_voltage,
        "crash_count": crash_count,
        "recovered_crash": recovered_crash,
        "ending_crash": crash_count - recovered_crash,
    }


def create_full_frame(width, height, frames, trackType='unknown'):
    track_obj = {
        "id": 0,
        "begin": 0,
        "end": frames - 1,
        "confidencePairs": [
            [
                trackType,
                1.0,
            ]
        ],
        "attributes": {},
        "features": [],
    }
    for frame in range(frames):
        frame = frame
        feature = {
            "frame": frame,
            "bounds": [0, 0, width, height],
            "attributes": {},
        }
        track_obj["begin"] = min(track_obj["begin"], frame)
        track_obj["end"] = max(track_obj["end"], frame)
        track_obj["features"].append(feature)
    return {
        "tracks": {"0": track_obj},
        "groups": {},
        "version": 2,
    }


def process_dive_dataset(gc: girder_client.GirderClient, folderId: str, frame_step=10):
    task_defaults = gc.get(f'dive_dataset/{folderId}/task-defaults')
    logging.info(f"Task Defaults: {task_defaults}")
    videoId = task_defaults.get('video', {}).get('fileId', False)
    if videoId:
        videoName = task_defaults.get('video', {}).get('filename', 'default.mp4')
        with tempfile.TemporaryDirectory() as _working_directory:
            _working_directory_path = Path(_working_directory)
            file_name = str(_working_directory_path / videoName)
            logging.info(f"Processing Video: {videoName}")
            gc.downloadFile(videoId, file_name)
            data = process_video(file_name, frame_step)
            trackJSON = data['track_json']
            outputFileName = './output.json'
            with open(outputFileName, 'w') as annotation_file:
                json.dump(trackJSON, annotation_file, separators=(',', ':'), sort_keys=False)
            gc.uploadFileToFolder(folderId, outputFileName)
            gc.post(f'dive_rpc/postprocess/{folderId}', data={"skipJobs": True})
            os.remove(outputFileName)
        return data


def process_metadata(args, gc: girder_client.GirderClient, stats):
    DIVEMetadataId = args.DIVEMetadata
    DIVEDatasetId = args.DIVEDirectory.split('/')[-2]

    DIVEMetadataRoot = args.DIVEMetadataRoot
    if not DIVEMetadataRoot:
        logging.warning('DIVE Metadata is not included, skipping setting values')
        return
    MetadataKey = args.MetadataKey
    MetadataValue = args.MetadataValue

    # First we check the root filter to see if there is a MetadataKey that exists
    current_filter_values = gc.get(f'/dive_metadata/{DIVEMetadataRoot}/metadata_keys')
    logging.info(f'Current Filter Values: {current_filter_values}')
    logging.info(f'MetadataRoot: {DIVEMetadataRoot} DatasetId: {DIVEDatasetId} MetadataKey: {MetadataKey} MetadataValue: {MetadataValue}')
    logging.info(f'Stats: {stats}')
    add_new_metadata(gc, DIVEMetadataRoot, DIVEDatasetId, current_filter_values, 'Average Throttle', stats['avg_throttle'], 'numerical',False)
    add_new_metadata(gc, DIVEMetadataRoot, DIVEDatasetId, current_filter_values, 'Average Battery', stats['avg_voltage'], 'numerical',False)
    add_new_metadata(gc, DIVEMetadataRoot, DIVEDatasetId, current_filter_values, 'Total Crashes', stats['crash_count'], 'numerical',False)
    add_new_metadata(gc, DIVEMetadataRoot, DIVEDatasetId, current_filter_values, 'Recovered Crashes', stats['recovered_crash'], 'numerical',False)
    add_new_metadata(gc, DIVEMetadataRoot, DIVEDatasetId, current_filter_values, 'Ending Crashes', stats['ending_crash'], 'numerical',False)


def add_new_metadata(gc: girder_client.GirderClient, DIVEMetadataRoot, DIVEDatasetId, current_values, key, value, category = 'search', unlocked = False):
    if key not in current_values['metadataKeys'].keys():
        logging.info('Adding the new key to MetadataRoot')
        # Field should be unlocked if the user who is running the task is not the owner.  Only owners can add new data to fields.
        gc.put(f'dive_metadata/{DIVEMetadataRoot}/add_key', {"key": key, "value": value, "category": "search", "unlocked": unlocked})

        # If we want the new Metadata Item to not be under the Advanced section we should add it to the main Display
        root_data = gc.get(f'folder/{DIVEMetadataRoot}')
        diveMetadataFilter = root_data['meta'].get('DIVEMetadataFilter', False)
        if diveMetadataFilter:
            diveMetadataFilter['display'].append(key)
            logging.info(diveMetadataFilter)
            gc.put(f'/folder/{DIVEMetadataRoot}/metadata', json={'DIVEMetadataFilter': diveMetadataFilter})
    # Now we set the actual value to the system
    gc.patch(f'dive_metadata/{DIVEDatasetId}', {"key": key, "value": value})


def main(args):

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    logging.info('\n>> CLI Parameters ...\n')
    logging.info(vars(args))
    process_input_args(args, gc)


def load_templates(paths):
    templates = []
    for path in paths:
        path = str(path)
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

def create_full_track():
    return {
        "id": 0,
        "begin": 0,
        "end": 0,
        "confidencePairs": [["unknown", 1.0]],
        "meta": {"time": True},
        "attributes": {},
        "features": [],
    }



if __name__ == '__main__':

    main(CLIArgumentParser().parse_args())