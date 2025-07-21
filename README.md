
# Document Scanner Scripts

This project includes Python scripts for scanning documents from images using OpenCV. The scripts detect document edges, apply perspective transformation, and output a cleaned-up scanned version of the image.

---

## Dependencies

Install Python dependencies via pip:

```
pip install -r requirements.txt
```

### requirements.txt Content:

```
opencv-python
numpy
imutils
matplotlib
scikit-image
```

---

## Scripts Overview

1. **scan.py**

   - Purpose: Process a **single image** and output a perspective-corrected scan.
   - Output: Saves the scanned document as `scanned_document.jpg`.

2. **scandebug.py**

   - Purpose: Same as `scan.py` but includes step-by-step visualization using Matplotlib.
   - Steps displayed:
     - Original image
     - Edge detection (Canny)
     - Document contour detection
     - Final perspective transform

3. **Scanall.py**

    - Purpose: Process an **entire folder** and output a perspective-corrected scan.
    - Output: Saves the scanned documents in specified folder path default: `scanned_outputs` as `scanned_{orignalfilename}.jpg`. (creating folder if missing)
     - pulls documents from any file path but default: `input_images`
---

## Usage Instructions

### To scan a single image:

```
python scan.py
```

### To debug with visualization of steps:

```
python scandebug.py
```

### To scan folder of images:

```
python scanall.py
```
---

## Project Structure

```
/project-root
    scan.py
    scandebug.py
    scanall.py
    requirements.txt
    README.txt
    input_images/ (optional for batch processing)
```

---

## Customization

### 📂 Input & Output Paths
By default, the script prompts for an input folder or accepts it via command-line. The output folder is created inside the input folder.
```bash
import sys

# === Config: Input Folder via CLI or Prompt ===
if len(sys.argv) > 1:
    input_folder = sys.argv[1]
else:
    input_folder = input("Enter input folder path: ")

input_folder = os.path.abspath(input_folder)
output_folder = os.path.join(input_folder, "scanned_outputs")
os.makedirs(output_folder, exist_ok=True)

print(f"Input folder: {input_folder}")
print(f"Output folder: {output_folder}")
```

Replace lines 5-10 with about to use command as shown
```
python scanall.py
```
- User will be prompted for input folder path
---
Or
---
```
python scanall.py ./relative/path/to/images
```
with this customization the code will run in either configuration from the cli

---

## References

- OpenCV: [https://opencv.org/](https://opencv.org/)
- imutils: [https://github.com/jrosebr1/imutils](https://github.com/jrosebr1/imutils)
- scikit-image: [https://scikit-image.org/](https://scikit-image.org/)

---


