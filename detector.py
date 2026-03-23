from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

print("Loading YOLO model...")
yolo_model = YOLO('yolov8n.pt')
print("YOLO ready")

YOLO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

def enhance_image(image):
    image = image.filter(ImageFilter.SHARPEN)
    image = ImageEnhance.Contrast(image).enhance(1.3)
    image = ImageEnhance.Brightness(image).enhance(1.1)
    return image

def is_boring_region(crop):
    img_array = np.array(crop.convert('RGB'))
    std = np.std(img_array)
    mean = np.mean(img_array)
    return std < 25 or mean < 20

def augment_image(image):
    augmented = [image]
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    augmented.append(image.rotate(10, expand=False))
    augmented.append(image.rotate(-10, expand=False))
    augmented.append(ImageEnhance.Brightness(image).enhance(1.3))
    augmented.append(ImageEnhance.Brightness(image).enhance(0.7))
    augmented.append(ImageEnhance.Contrast(image).enhance(1.4))
    augmented.append(image.convert('L').convert('RGB'))
    return augmented

def get_contour_regions(image):
    img_array = np.array(image.convert('RGB'))
    width, height = image.size
    img_area = width * height

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = np.ones((6, 6), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    seen_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.04 or area > img_area * 0.60:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w > width * 0.85 or h > height * 0.85:
            continue

        aspect = w / max(h, 1)
        if aspect > 4 or aspect < 0.25:
            continue

        is_duplicate = False
        for sx, sy, sw, sh in seen_boxes:
            overlap_x = max(0, min(x+w, sx+sw) - max(x, sx))
            overlap_y = max(0, min(y+h, sy+sh) - max(y, sy))
            overlap = overlap_x * overlap_y
            if overlap > area * 0.5:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        cropped = image.crop((x1, y1, x2, y2))
        if not is_boring_region(cropped):
            regions.append({
                'crop': cropped,
                'box': (x1, y1, x2, y2),
                'source': 'contour',
                'yolo_label': None,
                'yolo_confidence': 0
            })
            seen_boxes.append((x, y, w, h))

    return regions

# CHANGED: Accept PIL Image directly and added fast_mode
def detect_regions(image_input, fast_mode=False):
    # Determine if input is a file path or a PIL image directly
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')
        
    width, height = image.size

    # YOLO natively accepts PIL Images, saving us disk I/O latency
    results = yolo_model(image, verbose=False, show=False)
    regions = []

    # Step 1 — YOLO with direct labels (Always runs)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            if confidence > 0.3:
                cropped = image.crop((x1, y1, x2, y2))
                if not is_boring_region(cropped):
                    yolo_label = YOLO_CLASSES.get(cls_id, None)
                    regions.append({
                        'crop': cropped,
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'source': 'yolo',
                        'yolo_label': yolo_label,
                        'yolo_confidence': confidence
                    })

    # Step 2 & 3 — Only run heavy contour/grid scans if NOT in fast_mode
    if not fast_mode:
        contour_regions = get_contour_regions(image)
        regions.extend(contour_regions)

        min_size = min(width, height) // 3
        for scale in [0.5, 0.75]:
            grid_w = max(int(width * scale), min_size)
            grid_h = max(int(height * scale), min_size)
            stride_w = grid_w // 2
            stride_h = grid_h // 2
            for y in range(0, height - grid_h, stride_h):
                for x in range(0, width - grid_w, stride_w):
                    cropped = image.crop((x, y, x + grid_w, y + grid_h))
                    if not is_boring_region(cropped):
                        regions.append({
                            'crop': cropped,
                            'box': (x, y, x + grid_w, y + grid_h),
                            'source': 'grid',
                            'yolo_label': None,
                            'yolo_confidence': 0
                        })

    return image, regions