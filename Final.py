import cv2 as cv
import numpy as np
import itertools
import os

# -------------------------------
# IMAGE ENHANCEMENT FUNCTIONS
# -------------------------------

def white_bal(img):
    img_float = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b, scale_g, scale_r = avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r
    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r
    return np.clip(img_float, 0, 255).astype(np.uint8)

def apply_clahe(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv.merge((l2, a, b))
    return cv.cvtColor(merged, cv.COLOR_LAB2BGR)

def dehazing(img, t_min=0.1, w=0.95):
    dark = cv.min(cv.min(img[:,:,0], img[:,:,1]), img[:,:,2])
    a = np.max(dark)
    transmission = 1 - w * (dark / (a + 1e-6))
    transmission = np.clip(transmission, t_min, 1)
    result = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        result[:,:,c] = (img[:,:,c] - (1 - transmission) * a) / transmission
    return np.clip(result, 0, 255).astype(np.uint8)

def gamma_correction(img, gamma=1.5):
    normalized = img / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.uint8(corrected * 255)

# -------------------------------
# ANALYSIS FUNCTIONS
# -------------------------------

def analyze_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    color_std = np.std(np.mean(img, axis=(0, 1)))
    haze = np.mean(np.min(img, axis=2))
    return {"brightness": brightness, "contrast": contrast, "color_std": color_std, "haze": haze}

def score_image(metrics):
    return (
        metrics["contrast"]
        + (255 - abs(127 - metrics["brightness"])) * 0.5
        - metrics["haze"] * 0.3
        + metrics["color_std"] * 0.4
    )

# -------------------------------
# DYNAMIC ENHANCER SELECTION
# -------------------------------

def dynamic_best_enhancer(img):
    """Tests all enhancer combos (1 to 3 enhancers) and returns best combo + result"""
    enhancers = {
        "CLAHE": apply_clahe,
        "White Balance": white_bal,
        "Dehazing": dehazing,
        "Gamma 1.6": lambda x: gamma_correction(x, 1.6),
        "Gamma 0.8": lambda x: gamma_correction(x, 0.8)
    }

    best_score = score_image(analyze_image(img))
    best_result = img
    best_combo = ["Original"]

    enhancer_names = list(enhancers.keys())

    for r in range(1, 4):
        for combo in itertools.permutations(enhancer_names, r):
            temp = img.copy()
            try:
                for name in combo:
                    temp = enhancers[name](temp)
                score = score_image(analyze_image(temp))
                if score > best_score:
                    best_score = score
                    best_result = temp
                    best_combo = combo
            except Exception:
                continue  

    return best_result, best_combo, best_score

# -------------------------------
# VIDEO PROCESSING
# -------------------------------

def process_video(path):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print(" Could not open video.")
        return

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_dynamic = cv.VideoWriter('enhanced_dynamic_video.mp4', fourcc, fps, (width, height))
    out_fixed = cv.VideoWriter('enhanced_best_combo_video.mp4', fourcc, fps, (width, height))

    frame_count = 0
    enhancers = {
        "CLAHE": apply_clahe,
        "White Balance": white_bal,
        "Dehazing": dehazing,
        "Gamma 1.6": lambda x: gamma_correction(x, 1.6),
        "Gamma 0.8": lambda x: gamma_correction(x, 0.8)
    }

    print("🔹 Evaluating best overall combo for full video...")
    sample_frames = []
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    for i in range(0, total_frames, max(1, total_frames // 10)):
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            small = cv.resize(frame, (0,0), fx=0.3, fy=0.3)
            sample_frames.append(small)

    best_score = -1
    best_combo_overall = ["CLAHE"]
    for frame in sample_frames:
        _, combo, score = dynamic_best_enhancer(frame)
        if score > best_score:
            best_score = score
            best_combo_overall = combo
    print(f"Overall Best Combo for video: {best_combo_overall}")

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    dynamic_best_combo = ["CLAHE"]

    print("Processing video frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:
            small = cv.resize(frame, (0,0), fx=0.3, fy=0.3)
            _, combo, score = dynamic_best_enhancer(small)
            dynamic_best_combo = combo
            print(f"Frame {frame_count}: Dynamic Combo = {combo}, Score={score:.2f}")

        enhanced_dynamic = frame.copy()
        for name in dynamic_best_combo:
            enhanced_dynamic = enhancers[name](enhanced_dynamic)

        enhanced_fixed = frame.copy()
        for name in best_combo_overall:
            enhanced_fixed = enhancers[name](enhanced_fixed)

        out_dynamic.write(enhanced_dynamic)
        out_fixed.write(enhanced_fixed)

        cv.imshow("Dynamic Enhancement", enhanced_dynamic)
        cv.imshow("Fixed Best Combo", enhanced_fixed)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out_dynamic.release()
    out_fixed.release()
    cv.destroyAllWindows()
    print("\n Saved videos:")
    print(" - enhanced_dynamic_video.mp4  (best combo updated dynamically)")
    print(" - enhanced_best_combo_video.mp4  (single best combo applied to all frames)")

# -------------------------------
# IMAGE PROCESSING
# -------------------------------

def process_image(path):
    img = cv.imread(path)
    if img is None:
        print(" Could not read image.")
        return
    best_img, combo, score = dynamic_best_enhancer(img)
    print(f"Best Combo: {combo} | Score: {score:.2f}")
    cv.imshow(f"Best Combo: {', '.join(combo)}", best_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# -------------------------------
# MAIN INPUT HANDLER
# -------------------------------

def process_input(path):
    if not os.path.exists(path):
        print(" File not found:", path)
        return

    ext = os.path.splitext(path)[-1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']

    if ext in image_exts:
        process_image(path)
    elif ext in video_exts:
        process_video(path)
    else:
        print(" Unsupported file type.")

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    path = input("Enter image or video path: ").strip()
    process_input(path)





def process_input(path):
    if not os.path.exists(path):
        print("File not found", path)
        return

    ext = os.path.splitext(path)[-1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']

    if ext in image_exts:
        process_image(path)
    elif ext in video_exts:
        process_video(path)
    else:
        print("FILE FORMAT NOT SUPPORTED.")



if __name__ == "__main__":
    path = input("Enter path of image or video: ").strip()
    process_input(path)

Gemini
Gemini in Drive doesn't support text files
Gemini in Workspace can make mistakes, so double-check responses. Learn more
