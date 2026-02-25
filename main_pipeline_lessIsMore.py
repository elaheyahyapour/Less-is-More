import os
import json
import random, re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from nuscenes.nuscenes import NuScenes
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from McNemar import mcnemar_from_two_dfs
from collections import Counter
from vlm_captioning   import init_blip2, generate_caption
from llm_inference    import init_llama, get_response
from prompt_builder import (
    build_structured_prompt_v2, build_unstructured_prompt_v2
)
from evaluate         import evaluate_action
from safety_metrics_helper import (safety_weighted_f1, expected_risk_cost, catastrophic_rates, compute_full_metrics, bootstrap_ci, _iou)



CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

YOLO_CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog'
}

_YOLO = None
def _init_yolo():
    global _YOLO
    if _YOLO is None:
        _YOLO = YOLO('yolov8x.pt')
    return _YOLO


_CAPTION_CACHE = {}
_DET_CACHE = {}

def detect_objects_yolo(image_path):
    if image_path in _DET_CACHE:
        return _DET_CACHE[image_path]

    yolo = _init_yolo()
    results = yolo(image_path, imgsz=1280, verbose=False)

    detections = []
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            cls  = int(b.cls[0].cpu().numpy())
            if cls in [0, 1, 2, 3, 5, 7, 9, 11]:
                if cls == 11:
                    thr = 0.20
                elif cls == 9:
                    thr = 0.35
                else:
                    thr = 0.50

                if conf >= thr:
                    detections.append({"bbox":[x1,y1,x2,y2], "confidence":conf, "class":cls})

    _DET_CACHE[image_path] = detections
    return detections

def _red_score_patch(patch_arr):
    r = patch_arr[...,0].astype(np.float32)
    g = patch_arr[...,1].astype(np.float32)
    b = patch_arr[...,2].astype(np.float32)
    return (r.mean()+1e-6)/((g.mean()+b.mean())+1e-6)

def add_red_light_flag(image_path, dets=None, r_over_gb=1.25, min_px=7):
    if dets is None:
        dets = detect_objects_yolo(image_path)
    try:
        im = Image.open(image_path).convert("RGB")
        W, H = im.size
    except Exception:
        return False
    votes = 0
    for d in dets:
        if d["class"] != 9:
            continue
        x1, y1, x2, y2 = map(int, d["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2-x1) < min_px or (y2-y1) < min_px:
            continue
        arr = np.asarray(im.crop((x1, y1, x2, y2)))
        r, g, b = arr[...,0].astype(np.float32), arr[...,1].astype(np.float32), arr[...,2].astype(np.float32)
        if (r.mean()+1e-6)/((g.mean()+b.mean())+1e-6) >= r_over_gb:
            votes += 1
    return votes > 0

def cached_caption(img_path, processor, blip2_model):
    if img_path in _CAPTION_CACHE:
        return _CAPTION_CACHE[img_path]
    cap = generate_caption(img_path, processor, blip2_model)
    _CAPTION_CACHE[img_path] = cap
    return cap
def infer_object_counts(image_path=None, dets=None):
    if dets is None:
        dets = detect_objects_yolo(image_path)
    counts = Counter(YOLO_CLASS_NAMES.get(d["class"], str(d["class"])) for d in dets)
    return dict(counts), dets

def _approx_min_distance_m(dets, img_h=900.0):

    if not dets:
        return None
    hs = [(d["bbox"][3] - d["bbox"][1]) for d in dets if (d["bbox"][3] - d["bbox"][1]) > 1]
    if not hs:
        return None
    distances = [1000.0 * (900.0 / float(img_h)) / h for h in hs]
    return float(min(distances))

def classify_complexity_per_frame(ttc, risk, det_count):

    score = 0
    score += 3 if ttc < 2.0 else (2 if ttc < 4.0 else 1)
    score += 3 if risk > 0.7 else (2 if risk > 0.4 else 1)
    score += 3 if det_count > 12 else (2 if det_count > 6 else 1)
    return "HIGH" if score >= 8 else ("MEDIUM" if score >= 5 else "LOW")

def format_latex_card(ttc, risk, obj_counts, complexity, action, reason):
    key = ["person", "car", "motorcycle", "truck", "bus", "traffic light", "stop sign"]
    filt = {k: obj_counts.get(k, 0) for k in key if obj_counts.get(k, 0) > 0}
    obj_lines = " \\\\ ".join(f"{k}: {v} detections" for k, v in filt.items()) or "none"

    return (
        "\\textbf{TTC \\& RISK ANALYSIS:}\\\\ "
        f"TTC: {ttc:.2f} seconds \\\\ "
        f"Risk Density: {risk:.2f} \\\\\n\n"
        "\\textbf{OBJECT DISTRIBUTION:}\\\\ "
        f"{obj_lines} \\\\\n\n"
        f"\\textbf{{SCENE COMPLEXITY:}} {complexity.title()} \\\\\n"
        "\\textbf{RECOMMENDATION:}\\\\ "
        f"{action}: {reason}"
    )

def detect_traffic_cues(image_path, dets=None, camera=None,
                        min_stop_rel_height=0.018, center_window=(0.20, 0.80)):
    if dets is None:
        dets = detect_objects_yolo(image_path)

    if camera and not camera.startswith("CAM_FRONT"):
        center_window = (0.05, 0.95)

    has_tl, has_stop, stop_close = False, False, False
    try:
        with Image.open(image_path) as im:
            W, H = im.width, im.height
            arr_full = np.asarray(im.convert("RGB"))
    except Exception:
        W, H = 1600, 900
        arr_full = None

    cx_lo, cx_hi = center_window
    for d in dets:
        if d["class"] == 9:
            has_tl = True
        if d["class"] == 11:
            x1, y1, x2, y2 = map(int, d["bbox"])
            rel_h = max(1.0, (y2 - y1)) / float(max(1, H))
            cx    = ((x1 + x2) / 2.0) / float(max(1, W))
            ar    = (x2 - x1) / max(1.0, (y2 - y1))
            red_ok = False
            if arr_full is not None and (x2 > x1) and (y2 > y1):
                patch = arr_full[y1:y2, x1:x2]
                red_ok = (_red_score_patch(patch) >= 1.25)

            shape_ok = (0.65 <= ar <= 1.5)
            color_ok = red_ok
            if (cx_lo <= cx <= cx_hi) and (rel_h >= min_stop_rel_height) and (shape_ok or color_ok):
                has_stop = True
                if rel_h >= (min_stop_rel_height + 0.015):
                    stop_close = True

    red = add_red_light_flag(image_path, dets)
    return {"stop_sign": has_stop, "stop_sign_close": stop_close, "traffic_light": has_tl, "red_light": red}





class SceneAnalysisLogger:
    def __init__(self, log_file="scene_analysis_log.txt"):
        self.log_file = log_file
        self.log_entries = []
        with open(self.log_file, 'w') as f:
            f.write(f"NuScenes Scene Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message, print_console=True):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_entries.append(log_entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        if print_console:
            print(message)

    def log_scene_analysis(self, scene_analysis):
        analysis_text = self._format_scene_analysis(scene_analysis)
        self.log(analysis_text, print_console=True)
        self.log("-" * 80, print_console=False)

    def _format_scene_analysis(self, analysis):
        output = []
        output.append(f" SCENE ANALYSIS EXPLANATION")
        output.append(f"Scene Token: {analysis['scene_token']}")
        output.append(f"Scene Name: {analysis['scene_name']}")

        lighting = analysis['lighting_stats']
        output.append(f" LIGHTING CONDITIONS:")
        output.append(f"   Total frames: {lighting['total_samples']}")
        output.append(f"   - Daytime frames: {lighting['day_samples']} ({lighting['day_percentage']:.1f}%)")
        output.append(f"   - Nighttime frames: {lighting['night_samples']} ({lighting['night_percentage']:.1f}%)")
        output.append(f"   - Hour distribution: {dict(lighting['hours_distribution'])}")

        output.append(f" TTC & RISK ANALYSIS:")
        output.append(f"   - Average TTC: {analysis['avg_ttc']:.2f} seconds")
        output.append(f"   - Average Risk Density: {analysis['avg_risk']:.2f}")
        output.append(f"   - Min TTC: {analysis['min_ttc']:.2f} seconds")
        output.append(f"   - Max Risk: {analysis['max_risk']:.2f}")

        if analysis['object_stats']:
            output.append(f" OBJECT DISTRIBUTION:")
            for obj_class, count in analysis['object_stats'].items():
                output.append(f"   - {obj_class}: {count} detections")

        complexity = analysis['complexity']
        complexity_emoji = "high" if complexity == "HIGH" else "medium" if complexity == "MEDIUM" else "low"
        output.append(f" SCENE COMPLEXITY: {complexity_emoji} {complexity}")

        output.append(f" RECOMMENDATION:")
        output.append(f"   {analysis['recommendation']}")

        if analysis.get('additional_metrics'):
            output.append(f" ADDITIONAL METRICS:")
            for metric, value in analysis['additional_metrics'].items():
                output.append(f"   - {metric}: {value}")

        return "\n".join(output)

logger = SceneAnalysisLogger()


def get_time_info_from_nuscenes(nusc, sample_data_token):
    sample_data = nusc.get('sample_data', sample_data_token)
    timestamp_us = sample_data['timestamp']
    timestamp_s = timestamp_us / 1_000_000
    dt_utc = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)

    filename = sample_data['filename']
    hour_from_filename = None
    try:
        parts = os.path.basename(filename).split('__')[0].split('-')
        if len(parts) >= 5:
            hour_from_filename = int(parts[4])
    except:
        hour_from_filename = None

    return {
        'timestamp_us': timestamp_us,
        'timestamp_s': timestamp_s,
        'datetime_utc': dt_utc,
        'hour_utc': dt_utc.hour,
        'hour_from_filename': hour_from_filename,
        'filename': filename
    }

def is_daytime_comprehensive(nusc, sample_data_token, method='filename'):
    time_info = get_time_info_from_nuscenes(nusc, sample_data_token)

    if method == 'filename':
        if time_info['hour_from_filename'] is not None:
            hour = time_info['hour_from_filename']
            return 6 <= hour < 18
        else:
            method = 'timestamp'

    if method == 'timestamp':
        hour = time_info['hour_utc']
        return 6 <= hour < 18

    if method == 'both':
        hour = time_info['hour_from_filename'] if time_info['hour_from_filename'] is not None else time_info['hour_utc']
        return 6 <= hour < 18

    return True

def analyze_scene_lighting_conditions(nusc, scene_token):
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']

    lighting_stats = {
        'total_samples': 0,
        'day_samples': 0,
        'night_samples': 0,
        'hours_distribution': {},
        'sample_times': []
    }

    while sample_token:
        sample = nusc.get('sample', sample_token)
        if 'CAM_FRONT' in sample['data']:
            cam_token = sample['data']['CAM_FRONT']
            time_info = get_time_info_from_nuscenes(nusc, cam_token)
            is_day = is_daytime_comprehensive(nusc, cam_token, method='both')

            lighting_stats['total_samples'] += 1
            if is_day:
                lighting_stats['day_samples'] += 1
            else:
                lighting_stats['night_samples'] += 1

            hour = time_info['hour_from_filename'] if time_info['hour_from_filename'] is not None else time_info['hour_utc']
            lighting_stats['hours_distribution'][hour] = lighting_stats['hours_distribution'].get(hour, 0) + 1
            lighting_stats['sample_times'].append({
                'sample_token': sample_token,
                'is_day': is_day,
                'hour': hour,
                'timestamp': time_info['timestamp_us']
            })

        sample_token = sample['next']

    total = lighting_stats['total_samples']
    lighting_stats['day_percentage'] = (lighting_stats['day_samples'] / total * 100) if total > 0 else 0
    lighting_stats['night_percentage'] = (lighting_stats['night_samples'] / total * 100) if total > 0 else 0
    return lighting_stats


def calculate_ttc_scale_based(current_bbox, previous_bbox, dt):
    S_current = current_bbox[3] - current_bbox[1]
    S_previous = previous_bbox[3] - previous_bbox[1]
    dS_dt = (S_current - S_previous) / dt
    if dS_dt > 0:
        return S_current / dS_dt
    else:
        return float('inf')

def calculate_risk_density_grid(objects_data, grid_size=100, cell_size=2):
    n_cells = int(grid_size / cell_size)
    risk_grid = np.zeros((n_cells, n_cells))
    lambda_spatial = 20.0
    tau_temporal = 5.0
    epsilon = 0.1

    for x, y, ttc, distance in objects_data:
        if not (-grid_size / 2 <= x <= grid_size / 2 and -grid_size / 2 <= y <= grid_size / 2):
            continue

        grid_x = int((x + grid_size / 2) / cell_size)
        grid_y = int((y + grid_size / 2) / cell_size)
        if not (0 <= grid_x < n_cells and 0 <= grid_y < n_cells):
            continue

        ttc_risk = 1.0 / max(ttc, epsilon)
        spatial_decay = np.exp(-distance / lambda_spatial)
        temporal_decay = np.exp(-ttc / tau_temporal)
        risk_value = ttc_risk * spatial_decay * temporal_decay
        sigma = 3.0 / cell_size

        for i in range(max(0, grid_x - 3), min(n_cells, grid_x + 4)):
            for j in range(max(0, grid_y - 3), min(n_cells, grid_y + 4)):
                gauss_weight = np.exp(-((i - grid_x) ** 2 + (j - grid_y) ** 2) / (2 * sigma ** 2))
                risk_grid[i, j] += risk_value * gauss_weight

    return risk_grid

class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.object_stats = Counter()

    def update(self, detections, threshold=50):
        matched_tracks = {}

        img_scale = 0.10 * np.median([abs(d['bbox'][2] - d['bbox'][0]) for d in detections]) if detections else 160.0
        adaptive_thr = max(80.0, img_scale)
        for detection in detections:
            bbox = detection['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            obj_class = detection['class']
            class_name = YOLO_CLASS_NAMES.get(obj_class, f'class_{obj_class}')
            self.object_stats[class_name] += 1

            best_match, best_score = None, -1.0
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                if track['class'] != obj_class:
                    continue
                last_bbox = track['bboxes'][-1]
                cx, cy = track['centers'][-1]
                dist = np.hypot(center[0] - cx, center[1] - cy)
                iou = _iou(bbox, last_bbox)
                if (iou >= 0.1) or (dist <= adaptive_thr):
                    score = iou - 0.0005 * dist
                    if score > best_score:
                        best_score = score
                        best_match = track_id

            if best_match is not None:
                self.tracks[best_match]['bboxes'].append(bbox)
                self.tracks[best_match]['centers'].append(center)
                matched_tracks[best_match] = True
            else:
                self.tracks[self.next_id] = {'bboxes': [bbox], 'centers': [center], 'class': obj_class}
                self.next_id += 1

        tracks_to_remove = []
        for track_id, tr in list(self.tracks.items()):
            if track_id not in matched_tracks and len(tr['bboxes']) > 20:
                tracks_to_remove.append(track_id)
        for tid in tracks_to_remove:
            del self.tracks[tid]
        return self.tracks


def calculate_enhanced_ttc_and_risk(image_items):

    tracker = SimpleTracker()
    frame_stats = []
    all_ttcs, all_risk = [], []

    base_w = base_h = None
    for p, _ in image_items:
        try:
            with Image.open(p) as im:
                base_w, base_h = im.width, im.height
            break
        except Exception:
            continue
    if base_w is None:
        base_w, base_h = 1600.0, 900.0

    img_center_x = float(base_w) / 2.0

    for i in range(1, len(image_items)):
        image_path, t_curr = image_items[i]
        _, t_prev = image_items[i - 1]
        dt = max(1e-3, float(t_curr) - float(t_prev))

        detections = detect_objects_yolo(image_path)
        tracks = tracker.update(detections)

        current_ttcs, objects_data = [], []
        for _, tr in tracks.items():
            if len(tr['bboxes']) >= 2:
                cur, prev = tr['bboxes'][-1], tr['bboxes'][-2]
                ttc = calculate_ttc_scale_based(cur, prev, dt)
                current_ttcs.append(ttc)

                cx = (cur[0] + cur[2]) / 2.0
                h = max(1.0, cur[3] - cur[1])
                dist = 1000.0 * (900.0 / float(base_h)) / h
                ego_x = (cx - img_center_x) * dist / 1000.0
                ego_y = dist
                objects_data.append((ego_x, ego_y, ttc, dist))

        if not objects_data:
            pseudo = []
            for d in detections:
                h = max(1.0, d['bbox'][3] - d['bbox'][1])
                dist = 1000.0 * (900.0 / float(base_h)) / h
                pseudo.append((0.0, dist, 4.0, dist))
            objects_data = pseudo

        risk_grid = calculate_risk_density_grid(objects_data) if objects_data else None
        risk = float(np.mean(risk_grid)) if risk_grid is not None else 0.0
        all_ttcs.extend([t for t in current_ttcs if np.isfinite(t)])
        all_risk.append(risk)
        frame_stats.append({'frame': i, 'ttcs': current_ttcs, 'risk': risk})

    valid_ttcs = [t for t in all_ttcs if np.isfinite(t)]
    return {
        'avg_ttc': np.mean(valid_ttcs) if valid_ttcs else float('inf'),
        'min_ttc': np.min(valid_ttcs) if valid_ttcs else float('inf'),
        'max_ttc': np.max(valid_ttcs) if valid_ttcs else float('inf'),
        'avg_risk': np.mean(all_risk) if all_risk else 0.0,
        'max_risk': np.max(all_risk) if all_risk else 0.0,
        'min_risk': np.min(all_risk) if all_risk else 0.0,
        'object_stats': dict(tracker.object_stats),
        'total_detections': sum(tracker.object_stats.values()),
        'frame_stats': frame_stats
    }



def determine_scene_complexity(stats):
    complexity_score = 0
    if stats['avg_ttc'] < 2.0: complexity_score += 3
    elif stats['avg_ttc'] < 4.0: complexity_score += 2
    else: complexity_score += 1

    if stats['avg_risk'] > 0.7: complexity_score += 3
    elif stats['avg_risk'] > 0.4: complexity_score += 2
    else: complexity_score += 1

    if stats['total_detections'] > 100: complexity_score += 3
    elif stats['total_detections'] > 50: complexity_score += 2
    else: complexity_score += 1

    if complexity_score >= 8: return "HIGH"
    if complexity_score >= 5: return "MEDIUM"
    return "LOW"

def generate_recommendation(stats, complexity):
    recs = []
    if stats['avg_ttc'] < 2.0:
        recs.append("Short TTC values suggest immediate threats. Focus on temporal reasoning and real-time collision avoidance.")
    if stats['avg_risk'] > 0.6:
        recs.append("High risk density detected. Implement enhanced safety protocols.")
    if complexity == "HIGH":
        recs.append("Scene complexity is HIGH. Consider multi-modal sensor fusion for better perception.")
    if stats['object_stats'].get('person', 0) > 5:
        recs.append("High pedestrian activity detected. Prioritize pedestrian safety algorithms.")
    if not recs:
        recs.append("Scene appears manageable. Continue with standard autonomous driving protocols.")
    return " ".join(recs)

def comprehensive_scene_analysis(nusc, scene_token, max_frames=20):
    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']

    logger.log(f"\n🚀 Starting comprehensive analysis for scene: {scene_name}")
    logger.log("📊 Analyzing lighting conditions...")
    lighting_stats = analyze_scene_lighting_conditions(nusc, scene_token)

    logger.log("🖼️ Collecting image frames...")
    image_items = []
    sample_token = scene['first_sample_token']
    frame_count = 0

    while sample_token and frame_count < max_frames:
        sample = nusc.get('sample', sample_token)
        if 'CAM_FRONT' in sample['data']:
            cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])
            if os.path.exists(img_path):
                image_items.append((img_path, cam_data['timestamp'] / 1e6))
                frame_count += 1
        sample_token = sample['next']

    if len(image_items) >= 2:
        ttc_risk_stats = calculate_enhanced_ttc_and_risk(image_items)
    else:
        logger.log("⚠️ Insufficient frames for TTC analysis, using default values...")
        ttc_risk_stats = {
            'avg_ttc': 3.0, 'min_ttc': 1.0, 'max_ttc': 5.0,
            'avg_risk': 0.3, 'max_risk': 0.5, 'min_risk': 0.1,
            'object_stats': {}, 'total_detections': 0, 'frame_stats': []
        }

    complexity = determine_scene_complexity(ttc_risk_stats)
    recommendation = generate_recommendation(ttc_risk_stats, complexity)

    analysis = {
        'scene_token': scene_token,
        'scene_name': scene_name,
        'lighting_stats': lighting_stats,
        'avg_ttc': ttc_risk_stats['avg_ttc'],
        'min_ttc': ttc_risk_stats['min_ttc'],
        'max_ttc': ttc_risk_stats['max_ttc'],
        'avg_risk': ttc_risk_stats['avg_risk'],
        'max_risk': ttc_risk_stats['max_risk'],
        'min_risk': ttc_risk_stats['min_risk'],
        'object_stats': ttc_risk_stats['object_stats'],
        'total_detections': ttc_risk_stats['total_detections'],
        'complexity': complexity,
        'recommendation': recommendation,
        'additional_metrics': {
        'frames_analyzed': len(image_items),
        'detection_density': ttc_risk_stats['total_detections'] / max(1, len(image_items)),
        'risk_variance': np.var([f['risk'] for f in ttc_risk_stats['frame_stats']]) if ttc_risk_stats['frame_stats'] else 0
    }

    }
    logger.log_scene_analysis(analysis)
    return analysis

def scene_matches_filters(nusc, scene, time_filter=None):
    matches_time = True
    if time_filter:
        try:
            first_sample = nusc.get('sample', scene['first_sample_token'])
            cam_front_token = first_sample['data']['CAM_FRONT']
            is_day = is_daytime_comprehensive(nusc, cam_front_token, method='both')
            if time_filter.lower() == "day":
                matches_time = is_day
            elif time_filter.lower() == "night":
                matches_time = not is_day
        except Exception:
            matches_time = True
    return matches_time



def make_balanced_eval(metadata, per_class=400):
    by = {"stop": [], "slow down": [], "proceed": []}
    for e in metadata:
        by[e["ground_truth"]].append(e)
    out = []
    for k in by:
        take = min(per_class, len(by[k]))
        if take > 0:
            out.extend(random.sample(by[k], take))
    random.shuffle(out)
    return out

def extract_and_split_metadata_with_comprehensive_analysis(
    nusc, output_dir, train_ratio=0.8, time_filter=None,
    analyze_scenes=True, max_scenes_to_analyze=5
):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    def _make_balanced_eval(metadata, per_class=300):
        from collections import defaultdict
        buckets = defaultdict(list)
        for e in metadata:
            buckets[e["ground_truth"]].append(e)
        out = []
        for k, rows in buckets.items():
            take = min(per_class, len(rows))
            if take > 0:
                out.extend(random.sample(rows, take))
        random.shuffle(out)
        return out

    def _derive_stop_close(image_path, dets=None, cx_min=0.30, cx_max=0.70, min_rel_h=0.06):
        try:
            with Image.open(image_path) as im:
                W, H = im.size
        except Exception:
            W, H = 1600, 900
        if dets is None:
            dets = detect_objects_yolo(image_path)
        for d in dets:
            if d["class"] == 11:
                x1, y1, x2, y2 = d["bbox"]
                rel_h = max(1.0, (y2 - y1)) / float(max(1, H))
                cx = ((x1 + x2) / 2.0) / float(max(1, W))
                if (cx_min <= cx <= cx_max) and (rel_h >= min_rel_h):
                    return True
        return False

    scenes = [s for s in nusc.scene if scene_matches_filters(nusc, s, time_filter)]
    if len(scenes) == 0:
        logger.log(" No scenes match the specified filters!")
        return

    logger.log(f" Found {len(scenes)} scenes matching filters")

    if analyze_scenes:
        logger.log(f" DETAILED SCENE ANALYSIS (analyzing {min(max_scenes_to_analyze, len(scenes))} scenes)")
        logger.log("=" * 80)
        analysis_scenes = random.sample(scenes, min(max_scenes_to_analyze, len(scenes)))
        scene_analyses = []
        for scene in analysis_scenes:
            analysis = comprehensive_scene_analysis(nusc, scene['token'])
            scene_analyses.append(analysis)
        logger.log(f" ANALYSIS SUMMARY:")
        avg_complexity_scores = Counter([a['complexity'] for a in scene_analyses])
        logger.log(f"   Complexity distribution: {dict(avg_complexity_scores)}")
        avg_ttc = np.mean([a['avg_ttc'] for a in scene_analyses if a['avg_ttc'] != float('inf')])
        avg_risk = np.mean([a['avg_risk'] for a in scene_analyses])
        logger.log(f"   Average TTC across scenes: {avg_ttc:.2f} seconds")
        logger.log(f"   Average Risk across scenes: {avg_risk:.2f}")

    random.shuffle(scenes)
    train_cutoff = int(len(scenes) * train_ratio)
    scene_splits = {"train": scenes[:train_cutoff], "val": scenes[train_cutoff:]}

    logger.log(f" METADATA EXTRACTION:")
    logger.log(f"   Train scenes: {len(scene_splits['train'])}")
    logger.log(f"   Validation scenes: {len(scene_splits['val'])}")

    for split_name, split_scenes in scene_splits.items():
        metadata = []
        logger.log(f" Processing {split_name} split...")

        for scene in tqdm(split_scenes, desc=f"Processing {split_name} scenes", leave=False):
            sample_token = scene['first_sample_token']
            scene_frames = {cam: [] for cam in CAMERAS}

            while sample_token:
                sample = nusc.get('sample', sample_token)
                for cam in CAMERAS:
                    cam_data_token = sample['data'].get(cam)
                    if not cam_data_token:
                        continue
                    cam_data = nusc.get('sample_data', cam_data_token)
                    src_path = os.path.join(nusc.dataroot, cam_data['filename'])
                    if os.path.exists(src_path):
                        time_info = get_time_info_from_nuscenes(nusc, cam_data_token)
                        is_day = is_daytime_comprehensive(nusc, cam_data_token, method='both')
                        scene_frames[cam].append({
                            'path': src_path,
                            'token': cam_data_token,
                            'timestamp': cam_data['timestamp'],
                            'time_info': time_info,
                            'is_day': is_day
                        })
                sample_token = sample['next'] if sample['next'] else None

            for cam in CAMERAS:
                if len(scene_frames[cam]) < 2:
                    continue

                image_items = [(f['path'], f['timestamp'] / 1e6) for f in scene_frames[cam]]
                image_items = image_items[:15]

                try:
                    stats = calculate_enhanced_ttc_and_risk(image_items)
                    clip_frame_stats = stats['frame_stats']
                    risk_vals = [fs['risk'] for fs in clip_frame_stats] if clip_frame_stats else []
                    p90 = np.percentile(risk_vals, 90) if risk_vals else 0.0
                except Exception as e:
                    logger.log(f" Error in TTC calculation: {e}")
                    clip_frame_stats = [{'ttcs': [], 'risk': 0.0} for _ in image_items]
                    p90 = 0.0

                mid_img_path = image_items[min(len(image_items) // 2, len(image_items) - 1)][0]
                mid_dets = detect_objects_yolo(mid_img_path)
                try:
                    with Image.open(mid_img_path) as _im:
                        H_mid = _im.height
                except Exception:
                    H_mid = 900
                saw_stop_sign = any(
                    d['class'] == 11 and ((d['bbox'][3] - d['bbox'][1]) / float(H_mid)) >= 0.035
                    for d in mid_dets
                )

                def label_from_frame_present(ttc_frame, risk_frame, cues_k, saw_stop, p90_val):
                    RSLOW = max(0.25, 0.70 * p90_val)
                    RSTOP = max(0.28, 0.75 * p90_val)

                    if cues_k.get("red_light") or cues_k.get("stop_sign_close"):
                        if not (np.isfinite(ttc_frame) and ttc_frame > 3.5 and risk_frame < 0.15):
                            return "stop"

                    if cues_k.get("stop_sign"):
                        if (not np.isfinite(ttc_frame)) or (ttc_frame <= 3.0) or (risk_frame >= RSLOW):
                            return "stop"

                    if saw_stop:
                        if (not np.isfinite(ttc_frame)) or (ttc_frame <= 3.0) or (risk_frame >= RSLOW):
                            return "stop"

                    if (np.isfinite(ttc_frame) and ttc_frame <= 1.7 and risk_frame >= RSTOP):
                        return "stop"
                    if (np.isfinite(ttc_frame) and ttc_frame <= 2.6) or (risk_frame >= RSLOW):
                        return "slow down"
                    return "proceed"

                scene_frames_clip = scene_frames[cam][:len(image_items)]
                num_valid_frames = min(len(scene_frames_clip) - 1, len(clip_frame_stats))
                for k in range(num_valid_frames):
                    frame = scene_frames_clip[k + 1]
                    fstat = clip_frame_stats[k]

                    ttc_frame = float(np.mean(fstat['ttcs'])) if fstat['ttcs'] else float('inf')
                    risk_frame = float(fstat['risk'])

                    dets_k = detect_objects_yolo(frame['path'])
                    cues_k = detect_traffic_cues(frame['path'], dets=dets_k, camera=cam) or {}

                    try:
                        red_flag = add_red_light_flag(frame['path'], dets_k)
                    except Exception:
                        red_flag = False
                    if "stop_sign_close" not in cues_k:
                        cues_k["stop_sign_close"] = _derive_stop_close(frame['path'], dets_k)
                    cues_k["red_light"] = bool(red_flag)
                    cues_k["stop_sign"] = bool(cues_k.get("stop_sign", False))

                    gt = label_from_frame_present(ttc_frame, risk_frame, cues_k, saw_stop_sign, p90)

                    dst_filename = f"{frame['token']}.jpg"
                    dst_path = os.path.join(output_dir, "images", dst_filename)
                    if not os.path.exists(dst_path):
                        try:
                            img = Image.open(frame['path']).convert("RGB")
                            img.save(dst_path)
                        except Exception:
                            continue

                    hour_info = frame['time_info']
                    hour_val = hour_info['hour_from_filename'] if hour_info['hour_from_filename'] is not None else hour_info['hour_utc']

                    metadata.append({
                        "image": f"images/{dst_filename}",
                        "camera": cam,
                        "ttc": round(ttc_frame if np.isfinite(ttc_frame) else 5.0, 2),
                        "risk": round(risk_frame, 2),
                        "time_of_day": "day" if frame['is_day'] else "night",
                        "hour": int(hour_val),
                        "timestamp": frame['timestamp'],
                        "scene_id": scene['token'],
                        "tags": ["urban"],
                        "ground_truth": gt
                    })

        output_file = os.path.join(output_dir, f"{split_name}_metadata.json")
        counts = Counter(e["ground_truth"] for e in metadata)
        if split_name == "val" and all(counts.get(c, 0) > 0 for c in ["stop", "slow down", "proceed"]):
            per_class = min(400, counts["stop"], counts["slow down"], counts["proceed"])
            to_write = _make_balanced_eval(metadata, per_class=per_class)
        else:
            to_write = metadata

        with open(output_file, "w") as f:
            json.dump(to_write, f, indent=2)

        logger.log(f" Saved {len(to_write)} entries to {output_file}")

    logger.log(f" PROCESSING COMPLETE!")
    logger.log(f" Output directory: {output_dir}")
    logger.log(f" Log file: {logger.log_file}")



def quick_sanity(path):
    D = json.load(open(path))
    from collections import Counter
    y = Counter(e["ground_truth"] for e in D)
    ttc_vals = [e["ttc"] for e in D]
    risk_vals = [e["risk"] for e in D]

    print("GT dist:", y)
    print("TTC  min/med/p90/max:", np.min(ttc_vals), np.median(ttc_vals), np.percentile(ttc_vals,90), np.max(ttc_vals))
    print("Risk min/med/p90/max:", np.min(risk_vals), np.median(risk_vals), np.percentile(risk_vals,90), np.max(risk_vals))


def evaluate_vlm_prompts(
    metadata_json: str,
    output_dir: str,
    llama_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    blip2_device: str = "cuda",
    label_field: str = "ground_truth",
    feature_sets=None,
    day_only=False
):


    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_json, 'r') as f:
        data = json.load(f)

    if day_only:
        data = [e for e in data if e.get("time_of_day") == "day"]



    import torch
    random.seed(456);
    np.random.seed(456);
    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.shuffle(data)


    natural = data[:]
    data = natural[:]
    print("\n=== PREFLIGHT ===")
    print("Total loaded:", len(data))
    print("Cameras:", Counter(e.get("camera", "?") for e in data))
    print("GT overall:", Counter(e["ground_truth"] for e in data))

    TARGET = 2000
    by_cls = defaultdict(list)
    for e in data:
        by_cls[e["ground_truth"]].append(e)

    per = min(len(by_cls["stop"]), len(by_cls["slow down"]), len(by_cls["proceed"]), TARGET // 3)
    data = (random.sample(by_cls["stop"], per) +
            random.sample(by_cls["slow down"], per) +
            random.sample(by_cls["proceed"], per))
    random.shuffle(data)
    print("Working set:", len(data), "| per-class:", per)

    risks_by_cam = defaultdict(list)
    for e in data:
        r = float(e.get("risk", 0.0))
        if r >= 0: risks_by_cam[e.get("camera", "?")].append(r)

    def guarded_p90(arr):
        arr = [x for x in arr if x > 1e-6]
        if not arr: return 0.15
        return max(float(np.percentile(arr, 90)), 0.15)

    risk_p90_by_cam = {cam: guarded_p90(vals) for cam, vals in risks_by_cam.items()}
    global_risk_p90 = guarded_p90([float(e.get("risk", 0.0)) for e in data])

    print("risk_p90_by_cam:", {k: round(v, 3) for k, v in risk_p90_by_cam.items()})
    print("global_risk_p90:", round(global_risk_p90, 3))

    ml_model = None
    try:
        n_train = max(300, int(0.6 * len(data)))
        ml_train = data[:n_train]
        ml_model = train_ml_baseline(ml_train, risk_p90_by_cam, global_risk_p90)
        print("[ML] Trained logistic-regression baseline on", len(ml_train), "samples")
    except Exception as _e:
        print("[ML] Training failed, will skip ML baseline:", _e)


    processor, blip2_model = init_blip2(device=blip2_device, use_device_map_auto=False)
    tok, mdl = init_llama(model_id=llama_model_id)




    def label_from_future_window(win_ttc, win_risk, saw_stop_sign_future, p90_future,
                                 ttc_stop=1.5, ttc_slow=2.5):
        RSTOP = max(0.60 * p90_future, 0.02)
        RSLOW = max(0.35 * p90_future, 0.01)
        if saw_stop_sign_future and not (win_ttc > 3.5 and win_risk < RSLOW):
            return "stop"
        if (np.isfinite(win_ttc) and win_ttc <= ttc_stop) and (win_risk >= RSTOP):
            return "stop"
        if (np.isfinite(win_ttc) and win_ttc <= ttc_slow) or (win_risk >= RSLOW):
            return "slow down"
        return "proceed"

    def calibrate_decision(pred, ttc, risk, cues, risk_p90, min_dist=None):
        risk_n = risk / (risk_p90 + 1e-6)

        TTC_STOP, TTC_SLOW = 2.8, 3.8
        RSTOP, RSLOW = 0.7, 0.4
        DIST_STOP, DIST_SD = 12.0, 18.0

        red = bool((cues or {}).get("red_light"))
        sgn = bool((cues or {}).get("stop_sign"))
        close = bool((cues or {}).get("stop_sign_close"))
        any_stop_cue = red or close or sgn

        if any_stop_cue and not (np.isfinite(ttc) and ttc > 3.8 and risk_n < 0.3):
            return "stop"
        if any_stop_cue and (min_dist is None):
            return "stop"

        if not any_stop_cue and (cues or {}).get("caption_stop_ref"):
            if (min_dist is None or min_dist <= 18.0) or (risk_n >= 0.6) or (np.isfinite(ttc) and ttc <= 3.2):
                return "stop"

        if min_dist is not None:
            if min_dist <= DIST_STOP:
                return "stop"
            if pred == "proceed" and min_dist <= DIST_SD:
                pred = "slow down"

        if pred in ("proceed", "slow down"):
            if (np.isfinite(ttc) and ttc <= TTC_STOP) or (risk_n >= RSTOP):
                return "stop"
            if (np.isfinite(ttc) and ttc <= TTC_SLOW) or (risk_n >= RSLOW):
                return "slow down"
            return "proceed"

        if pred == "stop" and not any_stop_cue:
            if (np.isfinite(ttc) and ttc > TTC_STOP) and (risk_n < RSLOW) and (min_dist is None or min_dist > DIST_SD):
                return "slow down"

        return pred


    EPS = 1e-6

    def norm_risk(risk, rp90):
        return risk / (rp90 + EPS)

    def predict_rule_only(ttc, risk, cues, rp90, min_dist=None):
        r_n = norm_risk(risk, rp90)
        TTC_STOP, TTC_SLOW = 2.8, 3.8
        RSTOP, RSLOW = 0.7, 0.4
        DIST_STOP, DIST_SD = 12.0, 18.0

        red = bool((cues or {}).get("red_light"))
        sgn = bool((cues or {}).get("stop_sign"))
        close = bool((cues or {}).get("stop_sign_close"))
        any_stop_cue = red or sgn or close

        if any_stop_cue and not (np.isfinite(ttc) and ttc > 3.8 and r_n < 0.3):
            return "stop"

        if min_dist is not None:
            if min_dist <= DIST_STOP:
                return "stop"

        if (np.isfinite(ttc) and ttc <= TTC_STOP) or (r_n >= RSTOP):
            return "stop"
        if (np.isfinite(ttc) and ttc <= TTC_SLOW) or (r_n >= RSLOW):
            return "slow down"

        if (min_dist is not None) and (min_dist <= DIST_SD):
            return "slow down"

        return "proceed"

    def predict_heuristic(ttc, risk, cues, rp90, min_dist=None,
                          a=1.0, b=1.0, c=0.6, bonus_red=0.8, bonus_stop=0.4,
                          theta_stop=1.35, theta_slow=0.65):

        r_n = norm_risk(risk, rp90)
        inv_ttc = 0.0 if not np.isfinite(ttc) else 1.0 / max(ttc, 0.1)
        inv_dist = 0.0 if (min_dist is None) else 1.0 / max(min_dist, 0.5)

        s = a * inv_ttc + b * r_n + c * inv_dist

        if cues:
            if cues.get("red_light"):      s += bonus_red
            if cues.get("stop_sign_close"):
                s += bonus_stop
            elif cues.get("stop_sign"):
                s += 0.25

        if s >= theta_stop:
            return "stop"
        if s >= theta_slow:
            return "slow down"
        return "proceed"

    from sklearn.linear_model import LogisticRegression

    ML_CLASSES = ["stop", "slow down", "proceed"]
    ML_CLASS_TO_ID = {c: i for i, c in enumerate(ML_CLASSES)}

    def _complexity_to_num(x):
        if x is None: return 1
        x = str(x).upper()
        return 0 if "LOW" in x else (1 if "MEDIUM" in x else 2)

    def build_feature_vector(entry, cues, rp90, min_dist, obj_counts, complexity):
        ttc = float(entry["ttc"])
        risk = float(entry["risk"])
        r_n = norm_risk(risk, rp90)
        inv_ttc = 0.0 if not np.isfinite(ttc) else 1.0 / max(ttc, 0.1)
        inv_dist = 0.0 if (min_dist is None) else 1.0 / max(min_dist, 0.5)

        k = obj_counts or {}
        person = int(k.get("person", 0))
        car = int(k.get("car", 0))
        truck = int(k.get("truck", 0))
        bus = int(k.get("bus", 0))

        c = cues or {}
        red = int(bool(c.get("red_light")))
        sgn = int(bool(c.get("stop_sign")))
        close = int(bool(c.get("stop_sign_close")))

        comp = _complexity_to_num(complexity)

        return np.array([
            ttc, r_n, inv_ttc, inv_dist,
            person, car, truck, bus,
            red, sgn, close,
            comp
        ], dtype=float)

    def train_ml_baseline(train_entries, rp90_by_cam, global_rp90):
        X, y = [], []
        for e in train_entries:
            cam = e.get("camera", "?")
            rp90 = rp90_by_cam.get(cam, global_rp90)

            img_path = e["image"] if os.path.isabs(e["image"]) else os.path.join(
                os.path.dirname(os.path.abspath(__file__)), e["image"])
            dets = detect_objects_yolo(img_path)
            obj_cnts, _ = infer_object_counts(dets=dets)
            cues = detect_traffic_cues(img_path, dets=dets, camera=cam)
            try:
                with Image.open(img_path) as im:
                    min_dist = _approx_min_distance_m(dets, img_h=im.height)
            except Exception:
                min_dist = _approx_min_distance_m(dets)

            x = build_feature_vector(e, cues, rp90, min_dist, obj_cnts,
                                     classify_complexity_per_frame(e["ttc"], e["risk"], sum(obj_cnts.values())))
            X.append(x)
            y.append(ML_CLASS_TO_ID.get(e["ground_truth"], 2))

        if not X:
            return None

        X = np.vstack(X);
        y = np.array(y, dtype=int)

        lr = LogisticRegression(max_iter=1000, class_weight={"stop": 3.0, "slow down": 1.5, "proceed": 1.0})
        lr = LogisticRegression(max_iter=1000, class_weight={0: 3.0, 1: 1.5, 2: 1.0})
        lr.fit(X, y)
        return lr

    def predict_ml(entry, model, rp90_by_cam, global_rp90):
        if model is None:
            return "slow down"
        cam = entry.get("camera", "?")
        rp90 = rp90_by_cam.get(cam, global_rp90)
        img_path = entry["image"] if os.path.isabs(entry["image"]) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), entry["image"])
        dets = detect_objects_yolo(img_path)
        obj_cnts, _ = infer_object_counts(dets=dets)
        cues = detect_traffic_cues(img_path, dets=dets, camera=cam)
        try:
            with Image.open(img_path) as im:
                min_dist = _approx_min_distance_m(dets, img_h=im.height)
        except Exception:
            min_dist = _approx_min_distance_m(dets)

        x = build_feature_vector(entry, cues, rp90, min_dist, obj_cnts,
                                 classify_complexity_per_frame(entry["ttc"], entry["risk"], sum(obj_cnts.values())))
        pred_id = int(model.predict(x.reshape(1, -1))[0])
        return ML_CLASSES[pred_id]

    if not feature_sets:
        feature_sets = [
            {"name": "A_unstructured_caption_only", "mode": "unstructured", "features": []},
            {"name": "B_struct_ttc_risk", "mode": "structured", "features": ["ttc", "risk"]},
            {"name": "C_struct_ttc_risk_dist_objdist", "mode": "structured",
             "features": ["ttc", "risk", "distance", "objdist"]},
            {"name": "D_struct_full", "mode": "structured",
             "features": ["ttc", "risk", "cues", "complexity", "distance", "objdist"]},


            {"name": "Z1_baseline_rule_only", "mode": "baseline_rule", "features": ["ttc", "risk", "cues", "distance"]},
            {"name": "Z2_baseline_heuristic", "mode": "baseline_heur", "features": ["ttc", "risk", "cues", "distance"]},
            {"name": "Z3_baseline_ml", "mode": "baseline_ml",
             "features": ["ttc", "risk", "cues", "distance", "objdist", "complexity"]},
        ]



    all_metrics = []

    eval_variants = [
        {"label": "auto_parse+calib_on", "parse_mode": "auto", "calibration": True},
        {"label": "strict_parse+calib_on", "parse_mode": "strict", "calibration": True},
    ]

    for fs in feature_sets:
        for variant in eval_variants:
            name = f"{fs['name']}__{variant['label']}"
            mode = fs["mode"]
            features = set(fs.get("features", []))
            rows = []

            for entry in tqdm(data, desc=f"Ablation {name}"):
                try:
                    img_path = entry["image"]
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(os.path.dirname(os.path.abspath(metadata_json)), img_path)

                    ttc = float(entry["ttc"])
                    risk = float(entry["risk"])
                    y_true = entry.get(label_field, entry.get("ground_truth"))
                    entry_cam = entry.get("camera", "?")
                    rp90 = risk_p90_by_cam.get(entry_cam, global_risk_p90)

                    dets = detect_objects_yolo(img_path)
                    obj_counts, _ = infer_object_counts(dets=dets)
                    cues = detect_traffic_cues(img_path, dets=dets, camera=entry_cam)
                    try:
                        with Image.open(img_path) as im:
                            min_dist = _approx_min_distance_m(dets, img_h=im.height)
                    except Exception:
                        min_dist = _approx_min_distance_m(dets)

                    complexity = classify_complexity_per_frame(ttc, risk, sum(obj_counts.values()))

                    if mode == "baseline_rule":
                        ev_pred = predict_rule_only(ttc, risk, cues, rp90, min_dist)
                        adj_pred = ev_pred
                        prompt, raw_resp = "[RULE-ONLY]", '{"action":"%s"}' % ev_pred

                    elif mode == "baseline_heur":
                        ev_pred = predict_heuristic(ttc, risk, cues, rp90, min_dist)
                        adj_pred = ev_pred
                        prompt, raw_resp = "[HEURISTIC]", '{"action":"%s"}' % ev_pred

                    elif mode == "baseline_ml":
                        ev_pred = predict_ml(entry, ml_model, risk_p90_by_cam, global_risk_p90)
                        adj_pred = ev_pred
                        prompt, raw_resp = "[ML-LR]", '{"action":"%s"}' % ev_pred

                    else:
                        caption = cached_caption(img_path, processor, blip2_model)
                        cues["caption_stop_ref"] = ("stop sign" in caption.lower())
                        if mode == "unstructured":
                            prompt = build_unstructured_prompt_v2(caption)
                        else:
                            prompt = build_structured_prompt_v2(
                                caption,
                                ttc if "ttc" in features else float("nan"),
                                risk if "risk" in features else float("nan"),
                                complexity if "complexity" in features else None,
                                min_dist if "distance" in features else None,
                                obj_counts if "objdist" in features else None,
                                cues if "cues" in features else {}
                            )
                        raw_resp = get_response(prompt, tok, mdl)

                        if variant["parse_mode"] == "strict":
                            ev_pred = parse_strict_only(raw_resp)
                            if ev_pred == "__invalid__":
                                ev_pred = "slow down"
                        else:
                            ev = evaluate_action(raw_resp, y_true)
                            ev_pred = ev["predicted"]

                        use_cal = variant["calibration"] and (
                                    ("ttc" in features) or ("risk" in features) or ("cues" in features))
                        adj_pred = calibrate_decision(ev_pred, ttc, risk, (cues if "cues" in features else {}), rp90,
                                                      (
                                                          min_dist if "distance" in features else None)) if use_cal else ev_pred

                    reason = f"TTC={ttc:.2f}, Rn={norm_risk(risk, rp90):.2f}, min_d={min_dist if min_dist else np.nan}"
                    latex_card = format_latex_card(ttc, risk, obj_counts, complexity, adj_pred.upper(), reason)

                    rows.append({
                        "image": img_path,
                        "time_of_day": entry.get("time_of_day"),
                        "prompt_type": mode,
                        "ablation": name,
                        "prompt": str(prompt).replace("\n", " "),
                        "response": raw_resp,
                        "ground_truth": y_true,
                        "predicted_raw": ev_pred,
                        "predicted": adj_pred,
                        "correct": (adj_pred == y_true),
                        "ttc": ttc,
                        "risk": risk,
                        "complexity": complexity,
                        "min_distance": min_dist,
                        "obj_counts": obj_counts,
                        "summary_latex": latex_card
                    })

                except Exception as ex:
                    print(f"[{name}] ⚠️ error: {ex}")
                    continue

            df = pd.DataFrame(rows)
            if "strict_parse" in name:
                df["predicted"] = df["predicted"].apply(
                    lambda x: x if x in {"stop", "slow down", "proceed"} else "slow down"
                )

            csv_path = os.path.join(output_dir, f"{name}_results.csv")
            df.to_csv(csv_path, index=False)



            if df.empty:
                with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
                    f.write(f"{name}\nNo samples\n")
                all_metrics.append((name, 0.0, 0.0, 0.0, 0.0))
                continue

            if not df.empty:
                print(name, "| n=", len(df),
                      "| GT:", Counter(df["ground_truth"]),
                      "| Pred:", Counter(df["predicted"]))

            base = compute_full_metrics(df, pred_col="predicted")
            cm = base["cm_counts"]
            stop_total = cm[0, :].sum()
            slow_total = cm[1, :].sum()

            stop_recall = (cm[0, 0] / stop_total) if stop_total else 0.0
            stop_to_proceed = (cm[0, 2] / stop_total) if stop_total else 0.0
            slow_to_proceed = (cm[1, 2] / slow_total) if slow_total else 0.0

            with open(os.path.join(output_dir, f"{name}_metrics.txt"), "w") as f:
                f.write(f"{name}\n")
                f.write(
                    f"Acc={base['acc']:.3f}  MacroF1={base['macro_f1']:.3f}  MacroPrec={base['macro_prec']:.3f}  MacroRec={base['macro_rec']:.3f}\n")
                f.write(
                    f"SafetyF1={base['safety_weighted_f1']:.3f}  ExpectedRiskCost={base['expected_risk_cost']:.3f}\n")
                f.write(f"Catastrophic (STOP→PROCEED): per-STOP={base['catastrophe']['missed_stop_per_stop']:.3f}, "
                        f"overall={base['catastrophe']['catastrophic_overall']:.3f}, "
                        f"near={base['catastrophe']['near_catastrophic_overall']:.3f}\n\n")
                f.write(f"STOP recall={stop_recall:.3f}  STOP→PROCEED rate={stop_to_proceed:.3f}\n")
                f.write(f"SLOW DOWN→PROCEED (near-miss) rate={slow_to_proceed:.3f}\n")

                for c, stats in base["per_class"].items():
                    f.write(
                        f"[{c}] P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f} (n={stats['support']})\n")

            np.savetxt(os.path.join(output_dir, f"{name}_cm_counts.csv"), base["cm_counts"], fmt="%d", delimiter=",")
            np.savetxt(os.path.join(output_dir, f"{name}_cm_rownorm.csv"), base["cm_row_norm"], fmt="%.4f",
                       delimiter=",")

            cis = bootstrap_ci(df, pred_col="predicted", B=1000)
            with open(os.path.join(output_dir, f"{name}_metrics.txt"), "a") as f:
                f.write(f"\n95% CIs (bootstrap, B=1000): Acc∈[{cis['acc_ci'][0]:.3f},{cis['acc_ci'][1]:.3f}] "
                        f" MacroF1∈[{cis['macro_f1_ci'][0]:.3f},{cis['macro_f1_ci'][1]:.3f}]\n")

            all_metrics.append((name, base["acc"], base["macro_prec"], base["macro_rec"], base["macro_f1"]))



    with open(os.path.join(output_dir, "ablation_summary.txt"), "w") as f:
        for (n, a, p, r, ff) in all_metrics:
            f.write(f"{n:>36}  Acc={a:.3f}  Prec={p:.3f}  Rec={r:.3f}  F1={ff:.3f}\n")

    track_csvs = [os.path.join(output_dir, f"{name}_results.csv") for (name, _, _, _, _) in all_metrics]
    track_names = [t.split("_results.csv")[0].split(os.sep)[-1] for t in track_csvs]
    pairs = []
    for i in range(len(track_csvs)):
        for j in range(i + 1, len(track_csvs)):
            dfA = pd.read_csv(track_csvs[i])
            dfB = pd.read_csv(track_csvs[j])
            merged = dfA.merge(dfB, on="image", suffixes=("_A", "_B"))
            res = mcnemar_from_two_dfs(
                merged.rename(columns={"ground_truth_A": "ground_truth", "predicted_A": "predicted"}),
                merged.rename(columns={"ground_truth_B": "ground_truth", "predicted_B": "predicted"}),
                track_names[i], track_names[j]
            )
            pairs.append(res)

    pd.DataFrame(data).to_csv(os.path.join(output_dir, "eval_index.csv"), index=False)

    with open(os.path.join(output_dir, "significance_tests.txt"), "w") as f:
        for (a, b, b01, c10, p) in pairs:
            f.write(f"McNemar {a} vs {b}: b={b01}, c={c10}, n={b01 + c10}, p={p:.4g}\n")


STRICT_TOKENS = {"stop", "slow down", "proceed"}
STRICT_JSON_RE = re.compile(r'"\s*action\s*"\s*:\s*"(stop|slow down|proceed)"', flags=re.I)

def parse_strict_only(raw_resp):
    m = STRICT_JSON_RE.search(raw_resp)
    if m: return m.group(1).lower()
    m2 = re.search(r'\bACTION\s*[:\-]\s*(stop|slow down|proceed)\b', raw_resp, flags=re.I)
    if m2: return m2.group(1).lower()
    return "slow down"


def analyze_dataset_statistics(metadata_file, log_to_file=True):
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        stats = {
            'total_images': len(metadata),
            'camera_distribution': Counter([item['camera'] for item in metadata]),
            'time_distribution': Counter([item['time_of_day'] for item in metadata]),
            'hour_distribution': Counter([item['hour'] for item in metadata]),
            'ttc_stats': {
                'mean': np.mean([item['ttc'] for item in metadata]),
                'median': np.median([item['ttc'] for item in metadata]),
                'min': np.min([item['ttc'] for item in metadata]),
                'max': np.max([item['ttc'] for item in metadata])
            },
            'risk_stats': {
                'mean': np.mean([item['risk'] for item in metadata]),
                'median': np.median([item['risk'] for item in metadata]),
                'min': np.min([item['risk'] for item in metadata]),
                'max': np.max([item['risk'] for item in metadata])
            },
            'ground_truth_distribution': Counter([item['ground_truth'] for item in metadata])
        }
        if log_to_file:
            logger.log(f"\ DATASET STATISTICS for {metadata_file}:")
            logger.log(f"   Total images: {stats['total_images']}")
            logger.log(f"   Camera distribution: {dict(stats['camera_distribution'])}")
            logger.log(f"   Time distribution: {dict(stats['time_distribution'])}")
            logger.log(f"   TTC statistics: {stats['ttc_stats']}")
            logger.log(f"   Risk statistics: {stats['risk_stats']}")
            logger.log(f"   Ground truth distribution: {dict(stats['ground_truth_distribution'])}")
        return stats
    except Exception as e:
        logger.log(f" Error analyzing dataset statistics: {e}")
        return None



def generate_analysis_report(output_dirs):
    logger.log(f" GENERATING COMPREHENSIVE ANALYSIS REPORT")
    logger.log("=" * 60)
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            logger.log(f" Analysis for: {output_dir}")
            train_file = os.path.join(output_dir, "train_metadata.json")



            if os.path.exists(train_file):
                logger.log(f"    Training Set:")
                _ = analyze_dataset_statistics(train_file)
            val_file = os.path.join(output_dir, "val_metadata.json")
            if os.path.exists(val_file):
                logger.log(f"    Validation Set:")
                _ = analyze_dataset_statistics(val_file)
        else:
            logger.log(f" Directory not found: {output_dir}")

def create_visualization_data(metadata_file, output_file):
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        viz_data = {
            'ttc_values': [item['ttc'] for item in metadata],
            'risk_values': [item['risk'] for item in metadata],
            'hours': [item['hour'] for item in metadata],
            'cameras': [item['camera'] for item in metadata],
            'time_of_day': [item['time_of_day'] for item in metadata]
        }
        with open(output_file, 'w') as f:
            json.dump(viz_data, f, indent=2)
        logger.log(f" Visualization data saved to: {output_file}")
    except Exception as e:
        logger.log(f" Error creating visualization data: {e}")



"""if __name__ == "__main__":
    EVAL_JSON = "val_metadata_anti_leak.clean.json"
    evaluate_vlm_prompts(
        metadata_json=EVAL_JSON,
        output_dir="vlm_eval_ttc_results",
        llama_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        blip2_device="cuda",
        label_field="ground_truth",
    )"""

if __name__ == "__main__":
    EVAL_JSON = "bdd10k_10k_metadata.json"
    evaluate_vlm_prompts(
        metadata_json=EVAL_JSON,
        output_dir="vlm_eval_bdd10k",
        llama_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        blip2_device="cuda",
        label_field="ground_truth",
        day_only=False,
    )
