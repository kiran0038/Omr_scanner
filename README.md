# Omr_scanner
import cv2
import numpy as np
import csv

# Load the OMR image
image = cv2.imread(r"A:\csv\image.jpg")

if image is None:
    print("❌ Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold (invert so filled = white)
_, thresh = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

bubbles = []

# Filter bubbles
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 20 < w < 60 and 20 < h < 60:  # bubble size range
        roi = thresh[y:y+h, x:x+w]
        white = cv2.countNonZero(roi)
        total = w * h
        fill_ratio = white / float(total)
        cx = x + w // 2
        cy = y + h // 2
        bubbles.append((cx, cy, fill_ratio, x, y, w, h))

if not bubbles:
    print("⚠️ No bubbles detected")
    exit()

# Sort bubbles top-to-bottom
bubbles.sort(key=lambda b: b[1])

# Split into left and right columns
mid_x = np.median([b[0] for b in bubbles])
left_bubbles = [b for b in bubbles if b[0] < mid_x]
right_bubbles = [b for b in bubbles if b[0] > mid_x]

# Sort each column top-to-bottom
left_bubbles.sort(key=lambda b: b[1])
right_bubbles.sort(key=lambda b: b[1])

def detect_digit(col_bubbles):
    detected = []
    for i, b in enumerate(col_bubbles):
        digit = i  # row index corresponds to digit (0–9)
        detected.append((digit, b[2], b))  # (digit, fill_ratio, bubble_data)
    # Pick the most filled bubble
    return max(detected, key=lambda d: d[1])

left_digit, _, left_b = detect_digit(left_bubbles)
right_digit, _, right_b = detect_digit(right_bubbles)

print(f"✅ Left column digit: {left_digit}")
print(f"✅ Right column digit: {right_digit}")
print(f"👉 Roll number: {left_digit}{right_digit}")

# Save to CSV
with open("omr_result.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Left Digit", "Right Digit", "Roll No"])
    writer.writerow([left_digit, right_digit, f"{left_digit}{right_digit}"])

print("📂 Result saved in omr_result.csv")

# ---- Debug: Draw detected bubbles ----
output = image.copy()

# left_b = (cx, cy, fill_ratio, x, y, w, h)
x, y, w, h = left_b[3], left_b[4], left_b[5], left_b[6]
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

x, y, w, h = right_b[3], right_b[4], right_b[5], right_b[6]
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite("omr_detected.jpg", output)
print("🖼️ Debug image saved as omr_detected.jpg")
