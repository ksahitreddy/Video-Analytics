import os
import json
import cv2
import math
import numpy as np
import shutil
import datetime
import asyncio
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from videodb import connect, SceneExtractionType
import insightface
from insightface.app import FaceAnalysis
import easyocr
from sklearn.cluster import KMeans
import chainlit as cl
import weaviate
import weaviate.classes.config as wvc
import weaviate.classes.query as wq
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================= CONFIGURATION =================
# Keys
os.environ["VIDEO_DB_API_KEY"] = "sk-kJlCLndfAnrwcppXMwPhEk3isgKJQBur8Wlz3hyRiQ4"
WEAVIATE_URL = "https://r1zl2imwtii4ysuqv4oypg.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "MVRSU0MwK0hJc2pQeFpvcV93N3VlcWJHb2lIY0VubFZ0NkxJaVlUSTJRanFNWkQvVmh3ZCtQdXVEbXJjPV92MjAw"
GROQ_API_KEY = "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"

# Settings
SAMPLE_FPS = 5
FACE_MATCH_THRESHOLD = 0.5
OCR_CONFIDENCE = 0.4
FRAMES_DIR = "frames_temp"
os.makedirs(FRAMES_DIR, exist_ok=True)

# ================= GLOBAL MODELS =================
print("⏳ Loading AI Models...")
yolo_model = YOLO("yolov8n.pt")
# InsightFace on CPU prevents CUDA crashes but is slow -> blocking issue.
# We will run this in a thread to fix the timeout.
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
ocr_reader = easyocr.Reader(['en'], gpu=True)
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Models Loaded.")

# ================= WEAVIATE HELPERS =================
def get_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": "none"}
    )

def init_schema():
    client = get_weaviate_client()
    
    if not client.collections.exists("VideoFrame"):
        client.collections.create(
            name="VideoFrame",
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                wvc.Property(name="video_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="timestamp_formatted", data_type=wvc.DataType.TEXT),
                wvc.Property(name="scene_context", data_type=wvc.DataType.TEXT),
                wvc.Property(name="objects_summary", data_type=wvc.DataType.TEXT),
                wvc.Property(name="search_text", data_type=wvc.DataType.TEXT)
            ]
        )

    if not client.collections.exists("VideoIdentity"):
        client.collections.create(
            name="VideoIdentity",
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                wvc.Property(name="video_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="identity_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="attributes", data_type=wvc.DataType.TEXT),
                wvc.Property(name="search_text", data_type=wvc.DataType.TEXT)
            ]
        )

    if not client.collections.exists("VideoLibrary"):
        client.collections.create(
            name="VideoLibrary",
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                wvc.Property(name="video_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="processed_date", data_type=wvc.DataType.DATE),
            ]
        )
    
    client.close()

def get_existing_videos():
    client = get_weaviate_client()
    try:
        lib = client.collections.get("VideoLibrary")
        response = lib.query.fetch_objects(limit=20, return_properties=["video_name"])
        videos = [o.properties["video_name"] for o in response.objects]
        return list(set(videos)) 
    except:
        return []
    finally:
        client.close()

def add_video_to_library(video_name):
    client = get_weaviate_client()
    lib = client.collections.get("VideoLibrary")
    exists = lib.query.fetch_objects(
        filters=wq.Filter.by_property("video_name").equal(video_name),
        limit=1
    )
    if not exists.objects:
        lib.data.insert({
            "video_name": video_name,
            "processed_date": datetime.datetime.now(datetime.timezone.utc)
        })
    client.close()

# ================= ANALYTICS LOGIC (SYNCHRONOUS) =================
# This function does the heavy lifting. We will call it in a thread.
def analyze_frame_sync(frame, ts_s, tracker):
    # 1. YOLO Detection
    results = yolo_model(frame, verbose=False)
    dets = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(float, r.xyxy[0])
        dets.append(([x1, y1, x2, y2], float(r.conf[0]), int(r.cls[0])))
    
    # 2. Tracking
    tracks = tracker.update_tracks(dets, frame=frame)
    frame_dets = []
    
    for t in tracks:
        if not t.is_confirmed(): continue
        tid = t.track_id
        gid = f"G{tid}"
        cls_id = getattr(t, "det_class", -1)
        cls_name = yolo_model.names.get(cls_id, "Unknown")
        
        ltrb = t.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Crop logic
        if (x2-x1) < 30 or (y2-y1) < 30: continue
        crop = frame[y1:y2, x1:x2]
        
        # Color & Identity placeholders
        color = get_dominant_color(crop)
        plate_text = None
        face_emb = None
        
        # Identity (Face)
        if cls_name == 'person':
            faces = face_app.get(crop)
            if faces:
                face_emb = faces[0].embedding
        
        # Plate (Vehicle)
        elif cls_id in [2, 3, 5, 7]:
            try:
                ocr = ocr_reader.readtext(crop)
                for _, txt, conf in ocr:
                    clean = "".join(e for e in txt if e.isalnum()).upper()
                    if conf > OCR_CONFIDENCE and len(clean) > 3:
                        plate_text = clean
                        break
            except: pass

        frame_dets.append({
            "class_name": cls_name,
            "global_id": gid,
            "bbox_center": ((x1+x2)//2, (y1+y2)//2),
            "license_plate": plate_text,
            "color": color,
            "face_emb": face_emb
        })
        
    return frame_dets

def get_dominant_color(image, k=1):
    if image.size == 0: return "Unknown"
    try:
        resized = cv2.resize(image, (64, 64))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).reshape((-1, 3))
        clt = KMeans(n_clusters=k, n_init=5)
        clt.fit(rgb_img)
        r, g, b = clt.cluster_centers_[0]
        if r > 200 and g > 200 and b > 200: return "White"
        if r < 50 and g < 50 and b < 50: return "Black"
        if r > 150 and g > 150 and b < 100: return "Yellow"
        if r > g + 40 and r > b + 40: return "Red"
        if g > r + 40 and g > b + 40: return "Green"
        if b > r + 40 and b > g + 40: return "Blue"
        return "Grey" if abs(r-g)<30 else "Neutral"
    except: return "Unknown"

def get_movement_status(curr, prev, time_diff):
    if not prev or time_diff == 0: return "Unknown"
    speed = math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2) / time_diff
    return "Stationary" if speed < 5 else "Running/Fast" if speed > 50 else "Walking"

def format_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

async def process_video_pipeline(video_path, status_msg):
    print(f"🎬 Processing {video_path}...")
    try:
        conn = connect()
        coll = conn.get_collection()
        vid = coll.upload(file_path=video_path)
        index_id = vid.index_scenes(
            extraction_type=SceneExtractionType.time_based,
            extraction_config={"time": 1, "frame_count": 5},
            prompt="Describe scene in detail."
        )
        raw_scenes = vid.get_scene_index(index_id)
        scenes_data = []
        iterable = getattr(raw_scenes, 'scenes', raw_scenes)
        if isinstance(iterable, list):
            for i in iterable:
                scenes_data.append({
                    "start": float(getattr(i, 'start', 0)),
                    "end": float(getattr(i, 'end', 0)),
                    "text": getattr(i, 'description', getattr(i, 'text', ''))
                })
    except Exception as e:
        print(f"VideoDB Error: {e}")
        scenes_data = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step = max(1, int(round(fps / SAMPLE_FPS)))
    
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    
    # State Memory
    frames_meta = []
    known_embs = []
    known_names = []
    next_pid = 1
    track_identities = {}
    track_plates = {}
    track_velocity = {}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- CRITICAL FIX: Explicit yield ---
        await asyncio.sleep(0)

        if frame_idx % sample_step == 0:
            if len(frames_meta) % 5 == 0:
                perc = int((frame_idx / total_frames) * 100)
                status_msg.content = f"⚙️ **Processing Video...**\nAnalysis: {perc}% Complete"
                await status_msg.update()

            ts_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # --- CRITICAL FIX: Run Heavy AI in Thread ---
            # This prevents the websocket from timing out during face/yolo inference
            frame_results = await cl.make_async(analyze_frame_sync)(frame, ts_s, tracker)
            
            # Process results (lightweight logic)
            processed_dets = []
            for d in frame_results:
                gid = d['global_id']
                final_name = d['class_name']
                
                # Update Velocity
                center = d['bbox_center']
                prev_c, prev_t = track_velocity.get(gid, (None, None))
                move = "Unknown"
                if prev_c: move = get_movement_status(center, prev_c, ts_s - prev_t)
                track_velocity[gid] = (center, ts_s)
                
                # Update Identity (Face)
                if d['face_emb'] is not None:
                    emb = d['face_emb']
                    best_score = -1
                    best_idx = -1
                    for idx, saved_emb in enumerate(known_embs):
                        score = np.dot(emb, saved_emb) / (np.linalg.norm(emb) * np.linalg.norm(saved_emb))
                        if score > best_score: best_score, best_idx = score, idx
                    
                    if best_score > FACE_MATCH_THRESHOLD:
                        final_name = known_names[best_idx]
                    else:
                        final_name = f"Person {next_pid}"
                        known_embs.append(emb)
                        known_names.append(final_name)
                        next_pid += 1
                    track_identities[gid] = final_name
                elif gid in track_identities:
                    final_name = track_identities[gid]
                
                # Update Plate
                plate = d['license_plate']
                if plate:
                    track_plates[gid] = plate
                elif gid in track_plates:
                    plate = track_plates[gid]

                processed_dets.append({
                    "class_name": final_name,
                    "global_id": gid,
                    "license_plate": plate,
                    "color": d['color'],
                    "movement": move
                })

            scene_text = ""
            for s in scenes_data:
                if s["start"] <= ts_s < s["end"]:
                    scene_text = s["text"]
                    break
            
            frames_meta.append({
                "timestamp_s": ts_s,
                "scene_text": scene_text,
                "detections": processed_dets
            })
            
        frame_idx += 1
    
    cap.release()
    identities = defaultdict(list)
    for f in frames_meta:
        for d in f["detections"]:
            key = d["class_name"]
            if d.get("license_plate"): key = f"Vehicle {d['license_plate']}"
            identities[key].append({
                "timestamp_s": f["timestamp_s"],
                "color": d["color"],
                "license_plate": d["license_plate"],
                "movement": d["movement"]
            })
    return {"frames_meta": frames_meta, "consolidated_identities": identities}

async def upload_to_weaviate(data, video_name, status_msg):
    status_msg.content = "☁️ **Uploading to Knowledge Base...**"
    await status_msg.update()
    client = get_weaviate_client()
    
    frames_col = client.collections.get("VideoFrame")
    identities_col = client.collections.get("VideoIdentity")
    
    frames_col.data.delete_many(where=wq.Filter.by_property("video_name").equal(video_name))
    identities_col.data.delete_many(where=wq.Filter.by_property("video_name").equal(video_name))
    
    with frames_col.batch.dynamic() as batch:
        for f in data['frames_meta']:
            ts_str = format_seconds(f['timestamp_s'])
            objs = []
            for d in f['detections']:
                detail = f"{d['class_name']}"
                if d['license_plate']: detail += f" (Plate: {d['license_plate']})"
                if d['color'] != "Unknown": detail += f" ({d['color']})"
                objs.append(detail)
            obj_txt = ", ".join(objs) if objs else "None"
            narrative = f"Time {ts_str}. Scene: {f['scene_text']}. Objects: {obj_txt}"
            
            batch.add_object(
                properties={
                    "video_name": video_name,
                    "timestamp_formatted": ts_str,
                    "scene_context": f['scene_text'],
                    "objects_summary": obj_txt,
                    "search_text": narrative
                },
                vector=embed_model.encode(narrative).tolist()
            )
            
    with identities_col.batch.dynamic() as batch:
        for name, sightings in data['consolidated_identities'].items():
            first = format_seconds(sightings[0]['timestamp_s'])
            last = format_seconds(sightings[-1]['timestamp_s'])
            colors = [s['color'] for s in sightings if s['color'] != 'Unknown']
            dom_color = max(set(colors), key=colors.count) if colors else "Unknown"
            plate = next((s['license_plate'] for s in sightings if s['license_plate']), "None")
            narrative = f"Identity {name}. Seen {first}-{last}. Color: {dom_color}. Plate: {plate}."
            
            batch.add_object(
                properties={
                    "video_name": video_name,
                    "identity_name": name,
                    "attributes": f"Color: {dom_color}, Plate: {plate}",
                    "search_text": narrative
                },
                vector=embed_model.encode(narrative).tolist()
            )
    
    client.close()
    add_video_to_library(video_name)

# ================= CHAINLIT UI FLOW =================
@cl.on_chat_start
async def start():
    init_schema()
    videos = get_existing_videos()
    
    actions = [cl.Action(name="upload_new", value="new", label="📤 Upload New Video", payload={})]
    for v in videos:
        actions.append(cl.Action(name="select_video", value=v, label=f"📁 {v}", payload={}))
    
    res = await cl.AskActionMessage(
        content="**Welcome to the Video Analytics Hub.**\nChoose an option:",
        actions=actions
    ).send()
    
    video_name = None
    
    if res:
        try:
            action_name = res.name
            action_value = res.value
        except AttributeError:
            action_name = res.get("name")
            action_value = res.get("value")

        if action_name == "upload_new":
            files = None
            while files == None:
                files = await cl.AskFileMessage(
                    content="Please upload a surveillance video (MP4/MOV).",
                    accept=["video/mp4", "video/mov"],
                    max_size_mb=100, timeout=180
                ).send()
            
            file = files[0]
            video_name = file.name
            
            video_path = f"input_{file.name}"
            with open(video_path, "wb") as f:
                with open(file.path, "rb") as source:
                    shutil.copyfileobj(source, f)
            
            msg = cl.Message(content="⚙️ **Starting Video Pipeline...**")
            await msg.send()
            
            json_data = await process_video_pipeline(video_path, msg)
            await upload_to_weaviate(json_data, video_name, msg)
            
        else:
            video_name = action_value
            await cl.Message(content=f"📂 **Loaded:** `{video_name}`\nData retrieved from Knowledge Base.").send()

    cl.user_session.set("video_name", video_name)
    
    if video_name:
        await cl.Message(content=f"✅ **Ready!** You are chatting about `{video_name}`.\nAsk questions like *'Who entered at 00:02?'* or *'Did a red car pass by?'*").send()
    else:
        await cl.Message(content="❌ No video selected. Please reload.").send()

@cl.on_message
async def main(message: cl.Message):
    video_name = cl.user_session.get("video_name")
    if not video_name:
        await cl.Message("Please restart and select a video.").send()
        return

    client = get_weaviate_client()
    col = client.collections.get("VideoFrame")
    
    vec = embed_model.encode(message.content).tolist()
    
    res = col.query.hybrid(
        query=message.content,
        vector=vec,
        limit=15,
        filters=wq.Filter.by_property("video_name").equal(video_name)
    )
    
    logs = []
    scenes = set()
    for o in res.objects:
        logs.append(f"[{o.properties['timestamp_formatted']}] {o.properties['objects_summary']}")
        scenes.add(f"[{o.properties['timestamp_formatted']}] {o.properties['scene_context']}")
    
    client.close()
    
    groq = Groq(api_key=GROQ_API_KEY)
    
    sys_prompt = (
        "You are a Forensic Video Analyst. Use these two sources:\n"
        "1. SCENE CONTEXT: Trust for general scene description and counts.\n"
        "2. ANALYTICS LOGS: Trust for specific IDs (Person 1), License Plates, and Colors.\n"
        "Protocol: If IDs change (G1 -> G5) but context implies same person, merge them in your answer."
    )
    
    user_prompt = (
        f"Question: {message.content}\n\n"
        f"SCENE CONTEXT:\n{chr(10).join(sorted(list(scenes)))}\n\n"
        f"ANALYTICS LOGS:\n{chr(10).join(sorted(logs))}"
    )
    
    stream = await cl.make_async(groq.chat.completions.create)(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":sys_prompt}, {"role":"user","content":user_prompt}],
        stream=True
    )
    
    resp_msg = cl.Message(content="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            await resp_msg.stream_token(chunk.choices[0].delta.content)
    
    await resp_msg.update()
