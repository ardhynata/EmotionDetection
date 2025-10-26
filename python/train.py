from ultralytics import YOLO, checks, hub
import multiprocessing

def main():
    # Environment check
    checks()

    # Login to Ultralytics Hub
    hub.login('<YOUR_ULTRALYTICS_HUB_TOKEN>')

    # Load model
    model = YOLO('https://hub.ultralytics.com/models/MJJbb2I08Y1O2If6OJQP')

    # GPU training on device 0, with workers disabled (Windows-safe)
    results = model.train(device=0, workers=0, resume=True)

if __name__ == '__main__':
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
