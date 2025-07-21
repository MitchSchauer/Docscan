import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

# === Step 0: Load Image ===
image_path = "20250719_080620.jpg"
image = cv2.imread(image_path)
assert image is not None, f"Image not found at {image_path}" # Replace with your image path

orig = image.copy()

# Format Preview
def show_image(title, image, cmap=None):
    plt.figure()
    plt.imshow(image if cmap is None else image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Show Original
show_image("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# === Step 1: Resize and Preprocessing ===
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)

# --- Preprocessing: Grayscale, Blur, Canny ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Show Edges
show_image("Step 1: Edge Detection", edged, cmap='gray')

# === Step 2: Find Contours ===
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    raise Exception("No document-like contour found.")

# Show Outline
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
show_image("Step 2: Contour Outline", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# === Step 3: Perspective Transform ===
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

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Show Transform
show_image("Step 3: Perspective Transform", cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# === Save Final Output ===
output_path = "scanned_document.jpg"
cv2.imwrite(output_path, warped)
print(f"Scanned image saved to {output_path}")
