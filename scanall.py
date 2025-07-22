import cv2
import numpy as np
import imutils
import os
## import pytesseract

# === Config: Input & Output Folders ===
input_folder = "input_images"
output_folder = "scanned_outputs"
os.makedirs(output_folder, exist_ok=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def scan_document(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_path}: could not load image.")
        return None

    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print(f"No document-like contour found in {image_path}. Skipping.")
        return None

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return warped

# === Process and save ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)
        print(f"Processing {image_path}...")

        scanned = scan_document(image_path)
        if scanned is not None:
            save_path = os.path.join(output_folder, f"scanned_{filename}")
            cv2.imwrite(save_path, scanned)
            print(f"Saved scanned document to {save_path}")
        # Optional OCR Processing Uncomment '##' sections

            # === Extract text from the scanned image ===
            ##text = pytesseract.image_to_string(scanned)

            # === Save the extracted text to a .txt file ===
            ##text_filename = os.path.splitext(filename)[0] + '.txt'
            ##text_path = os.path.join(output_folder, text_filename)

            ##with open(text_path, 'w', encoding='utf-8') as f:
            ##    f.write(text)

            ##print(f"Extracted text saved to {text_path}")
