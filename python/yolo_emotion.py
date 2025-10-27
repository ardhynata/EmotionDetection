import sys, json, os, base64, io, requests, threading
from PIL import Image
from ultralytics import YOLO

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


try:
    # === 1. Load trained model ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "emode1-train1.pt")
    model = YOLO(model_path)

    # === 2. Read image bytes from stdin ===
    for line in sys.stdin:
        line = line.strip()
        img_bytes = base64.b64decode(line)

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(json.dumps({"error": f"cannot open image: {e}"}), flush=True)
            continue

        # === 3. Resize image to 64x64 ===
        img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)

        # === 4. Run model prediction ===
        results = model(img_resized, verbose=False)
        probs = results[0].probs.data.cpu().numpy()
        class_names = results[0].names
        raw_result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        # === 5. Process result ===
        happy = raw_result.get("nostress", 0.0)
        sad = raw_result.get("stress", 0.0)
        sad_percent = sad * 100.0

        json_result = {
            "happy": round(happy, 2),
            "sad": round(sad, 2)
        }

        print(json.dumps(json_result), flush=True)

        # === 6. Send Telegram alert if stress detected ===
        if sad > 0.7:
            msg = f"⚠️ Mahesaca Alert:\n😟 Stress terdeteksi!\nAI Confidence: {sad_percent:.2f}%"
            send_telegram_message(msg, image_pil=img)

except Exception as e:
    print(json.dumps({"error": str(e)}))
