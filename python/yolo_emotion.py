import sys, json, os, base64, io
from PIL import Image
from ultralytics import YOLO

try:
    # === 1. Load trained model ===
    # Replace the path below with your trained model path
    # (e.g. 'runs/classify/train/weights/best.pt' or a local file)
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "emode1-train1.pt")
    model = YOLO(model_path)  # or "runs/classify/exp/weights/best.pt" depending on your folder

    # === 2. Read image bytes from stdin ===

    for line in sys.stdin:
        line = line.strip()
        img_bytes = base64.b64decode(line)
        
        # Convert bytes to PIL Image
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(json.dumps({"error": f"cannot open image: {e}"}), flush=True)
            continue

        results = model(img, verbose=False)
        probs = results[0].probs.data.cpu().numpy()
        class_names = results[0].names
        raw_result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        # === 5. Rename keys and round values ===
        happy = raw_result.get("nostress", 0.0)
        sad = raw_result.get("stress", 0.0)

        json_result = {
            "happy": round(happy, 2),
            "sad": round(sad, 2)
        }
        print(json.dumps(json_result), flush=True)

except Exception as e:
    print(json.dumps({"error": str(e)}))