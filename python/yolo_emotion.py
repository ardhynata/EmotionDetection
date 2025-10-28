import sys, json, os, base64, io, requests, threading
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# === TELEGRAM CONFIG ===
BOT_TOKEN = "8401117624:AAEl959Ef4bszTeH8LekDV5VH7Y-RMnlsg0"
CHAT_ID = "1550912667"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

def _send_telegram_message(text, image_pil=None):
    try:
        if image_pil is not None:
            buf = io.BytesIO()
            image_pil.save(buf, format="JPEG", quality=80)
            buf.seek(0)
            files = {"photo": ("snapshot.jpg", buf, "image/jpeg")}
            data = {"chat_id": CHAT_ID, "caption": text}
            requests.post(f"{BASE_URL}/sendPhoto", data=data, files=files, timeout=10)
    except Exception as e:
        print(json.dumps({"telegram_error": str(e)}), flush=True)

def send_telegram_message(text, image_pil=None):
    thread = threading.Thread(target=_send_telegram_message, args=(text, image_pil), daemon=True)
    thread.start()

# --- Initialize Mediapipe face detector ---
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

try:
    # === 1. Load trained model ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "emode1-train2.pt")
    model = YOLO(model_path)

    BATCH_SIZE = 50
    frame_buffer = []
    noface_count = 0
    last_face_frame = None  # 🟢 Store the most recent frame with a face

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        img_bytes = base64.b64decode(line)

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(json.dumps({"error": f"cannot open image: {e}"}), flush=True)
            continue

        # --- Convert PIL image to OpenCV for face detection ---
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detections = face_detector.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        face_detected = detections.detections is not None and len(detections.detections) > 0

        if face_detected:
            face = detections.detections[0].location_data.relative_bounding_box
            h, w, _ = img_cv.shape
            x1 = max(int(face.xmin * w), 0)
            y1 = max(int(face.ymin * h), 0)
            x2 = min(int((face.xmin + face.width) * w), w)
            y2 = min(int((face.ymin + face.height) * h), h)

            img_with_bbox = img_cv.copy()
            cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_with_bbox_pil = Image.fromarray(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))

            # Crop and prepare for inference
            cropped_cv = img_cv[y1:y2, x1:x2]
            img_cropped = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

            # Resize for model
            img_resized = img_cropped.resize((64, 64), Image.Resampling.LANCZOS)

            # Run model only if face detected
            results = model(img_resized, verbose=False)
            probs = results[0].probs.data.cpu().numpy()
            class_names = results[0].names
            raw_result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

            happy = raw_result.get("nostress", 0.0)
            sad = raw_result.get("stress", 0.0)

            # Confidence adjustment
            accuracy = 0.795
            uncertainty = 1 - accuracy
            happy = happy * (1 - uncertainty) + 0.5 * uncertainty
            sad = sad * (1 - uncertainty) + 0.5 * uncertainty
            total = happy + sad
            if total > 0:
                happy /= total
                sad /= total

            frame_buffer.append((img_with_bbox_pil, happy, sad))
            last_face_frame = img_with_bbox_pil  # 🟢 update most recent face frame
        else:
            # Skip model inference and count this frame
            noface_count += 1
            frame_buffer.append((img, None, None))

        # === Process when batch full ===
        if len(frame_buffer) >= BATCH_SIZE:
            noface_ratio = noface_count / BATCH_SIZE

            if noface_ratio > 0.5:
                print(json.dumps({"result": "no-face"}), flush=True)
            else:
                valid_frames = [(h, s) for _, h, s in frame_buffer if h is not None and s is not None]
                if not valid_frames:
                    print(json.dumps({"result": "no-face"}), flush=True)
                else:
                    avg_happy = sum([h for h, _ in valid_frames]) / len(valid_frames)
                    avg_sad = sum([s for _, s in valid_frames]) / len(valid_frames)
                    json_result = {
                        "happy": round(avg_happy, 4),
                        "sad": round(avg_sad, 4)
                    }
                    print(json.dumps(json_result), flush=True)

                    if avg_sad > 0.6 and last_face_frame is not None:
                        msg = f"⚠️ Mahesaca Alert:\n😟 Stress terdeteksi!\nAI Confidence (avg): {avg_sad * 100:.2f}%"
                        send_telegram_message(msg, image_pil=last_face_frame)

            # Reset for next batch
            frame_buffer.clear()
            noface_count = 0
            last_face_frame = None  # 🟢 reset after sending

except Exception as e:
    print(json.dumps({"error": str(e)}))
