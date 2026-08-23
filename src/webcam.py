import os
import sys
import time
import argparse
import cv2
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_device
from src.models import get_model

INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

webcam_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def parse_args():
    parser = argparse.ArgumentParser(description="Webcam Head Pose Estimation")
    parser.add_argument("--model", type=str, required=True, choices=["efficientnet", "mobilenet"])
    parser.add_argument("--camera-id", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()

    m = args.model.lower()
    model_label = "EfficientNetB0" if m == "efficientnet" else "MobileNetV3-Small"
    checkpoint_path = f"outputs/models/{m}_best.pth"

    if not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Checkpoint tidak ditemukan: {checkpoint_path}")
        sys.exit(1)

    model = get_model(args.model).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[Model] {model_label} dimuat dari {checkpoint_path}")

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print("[ERROR] Haar Cascade tidak ditemukan.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka kamera (ID={args.camera_id})")
        sys.exit(1)

    print("[Webcam] Kamera terbuka. Tekan 'q' untuk keluar.")
    print(f"[Webcam] Model: {model_label} | Device: {device}")

    fps_history = []
    frame_count = 0

    with torch.inference_mode():
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces_sorted[0]

                # Padding 20%
                pad_x = int(w * 0.20)
                pad_y = int(h * 0.20)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    tensor = webcam_transform(pil_img).unsqueeze(0).to(device)

                    out = model(tensor)
                    pitch = float(out[0, 0].item())
                    yaw = float(out[0, 1].item())
                    roll = float(out[0, 2].item())

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    lines = [
                        f"Pitch: {pitch:+.2f} deg",
                        f"Yaw:   {yaw:+.2f} deg",
                        f"Roll:  {roll:+.2f} deg",
                    ]
                    for i, line in enumerate(lines):
                        cv2.putText(frame, line, (x1, y1 - 10 - 22 * (len(lines) - 1 - i)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

            t_frame_end = time.perf_counter()
            elapsed = t_frame_end - t_frame_start
            fps_now = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_history.append(fps_now)
            if len(fps_history) > 30:
                fps_history.pop(0)
            fps_avg = sum(fps_history) / len(fps_history)

            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Model: {model_label}", (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow(f"Head Pose - {model_label}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[Webcam] Selesai. Total frame: {frame_count}")

if __name__ == "__main__":
    main()
