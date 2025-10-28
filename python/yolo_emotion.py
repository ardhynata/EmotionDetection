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
    """Internal: performs the actual sending."""
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
    """Non-blocking Telegram send using background thread."""
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
    results_buffer = []
    frame_count = 0

    for line in sys.stdin:
        line = line.strip()
        img_bytes = base64.b64decode(line)
        frame_count += 1

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(json.dumps({"error": f"cannot open image: {e}"}), flush=True)
            continue

        # --- Convert PIL image to OpenCV for face detection ---
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detections = face_detector.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        if detections.detections:
            face = detections.detections[0].location_data.relative_bounding_box
            h, w, _ = img_cv.shape
            x1 = max(int(face.xmin * w), 0)
            y1 = max(int(face.ymin * h), 0)
            x2 = min(int((face.xmin + face.width) * w), w)
            y2 = min(int((face.ymin + face.height) * h), h)

            img_with_bbox = img_cv.copy()
            cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_with_bbox_pil = Image.fromarray(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))

            cropped_cv = img_cv[y1:y2, x1:x2]
            img_cropped = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
        else:
            img_cropped = img
            img_with_bbox_pil = img

        # Resize for model
        img_resized = img_cropped.resize((64, 64), Image.Resampling.LANCZOS)

        # Run model
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

        # Append to buffer
        frame_buffer.append((img_with_bbox_pil, happy, sad))

        # When batch full:
        if len(frame_buffer) >= BATCH_SIZE:
            avg_happy = sum([h for _, h, _ in frame_buffer]) / BATCH_SIZE
            avg_sad = sum([s for _, _, s in frame_buffer]) / BATCH_SIZE

            # Prepare averaged result
            json_result = {
                "happy": round(avg_happy, 4),
                "sad": round(avg_sad, 4)
            }
            print(json.dumps(json_result), flush=True)

            # Optional: Send the 5th frame to Telegram (middle of batch)
            mid_frame, _, _ = frame_buffer[BATCH_SIZE // 2]
            if avg_sad > 0.6:
                msg = f"⚠️ Mahesaca Alert:\n😟 Stress terdeteksi!\nAI Confidence (avg): {avg_sad * 100:.2f}%"
                send_telegram_message(msg, image_pil=mid_frame)

            # Clear buffer for next batch
            frame_buffer.clear()

except Exception as e:
    print(json.dumps({"error": str(e)}))
