import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Standard ImageNet parameters used in research
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
FACE_PADDING_RATIO = 0.20

def get_val_transform() -> T.Compose:
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def detect_and_crop_face(image_rgb: np.ndarray, padding_ratio: float = FACE_PADDING_RATIO):
    """
    Detects face using Haar Cascade. 
    Returns:
        face_crop_rgb: cropped numpy array
        bbox: (x1, y1, x2, y2)
    Returns None, None if no face is detected.
    """
    # Use Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    # Get largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # Calculate padding
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    img_h, img_w = image_rgb.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    face_crop = image_rgb[y1:y2, x1:x2]
    return face_crop, (x1, y1, x2, y2)

def preprocess_image(image_rgb: np.ndarray, fallback_full_image: bool = False):
    """
    Preprocess image for inference.
    Returns:
        tensor: preprocessed batch tensor ready for model
        face_crop: the cropped numpy array (for display)
        status: "success", "no_face_fallback", or "no_face_failed"
    """
    face_crop, bbox = detect_and_crop_face(image_rgb)
    
    if face_crop is None:
        if fallback_full_image:
            face_crop = image_rgb
            status = "no_face_fallback"
        else:
            return None, None, "no_face_failed"
    else:
        status = "success"

    pil_img = Image.fromarray(face_crop)
    transform = get_val_transform()
    tensor = transform(pil_img).unsqueeze(0) # Add batch dim
    
    return tensor, face_crop, status
