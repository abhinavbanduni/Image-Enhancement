import cv2 as cv
import numpy as np
import os

# IMAGE ENHANCING FUNCTIONS

def white_bal(img):
    img_float = img.astype(np.float32)
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r
    balanced = np.clip(img_float, 0, 255).astype(np.uint8)
    return balanced

def apply_clahe(img):
    lab = cv.cvtColor(img , cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0 , tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    merged = cv.merge((l_clahe, a, b))
    final_img = cv.cvtColor(merged , cv.COLOR_LAB2BGR)
    return final_img

def dehazing(img , t_min=0.1 , w=0.95):
    dark = cv.min(cv.min(img[:,:,0], img[:,:,1]), img[:,:,2])
    a = np.max(dark)
    transmission = 1 - w*(dark/a)
    transmission = np.clip(transmission , t_min , 1)
    result = np.empty_like(img , dtype=np.float32)
    for c in range(3):
        result[:,:,c] = (img[:,:,c] - (1-transmission)*a)/ transmission
    result = np.clip(result,0,255).astype(np.uint8)
    return result

def gamma_correction(img, gamma=1.5):
    img_normalised = img / 255.0
    corrected = np.power(img_normalised, (1/gamma))
    return np.uint8(corrected * 255)

# ANALYZING FUNCTIONS

def analyze_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    contrast = np.std(gray)
    avg_color = np.mean(img, axis=(0, 1))
    color_std = np.std(avg_color)
    dark_channel = np.min(img, axis=2)
    haze_level = np.mean(dark_channel)

    return {
        "brightness": avg_brightness,
        "contrast": contrast,
        "color_std": color_std,
        "haze": haze_level
    }

# COMBINATION DECISION

def dynamic_enhance_logic(img):
    metrics = analyze_image(img)
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]
    color_std = metrics["color_std"]
    haze = metrics["haze"]

    print(f"Brightness: {brightness:.1f}, Contrast: {contrast:.1f}, Color Std: {color_std:.1f}, Haze: {haze:.1f}")

    applied = []
    enhanced = img.copy()

    # Apply based on multiple conditions
    if color_std > 25:
        enhanced = white_bal(enhanced)
        applied.append("White Balance")

    if haze > 100:
        enhanced = dehazing(enhanced)
        applied.append("Dehazing")

    if contrast < 40:
        enhanced = apply_clahe(enhanced)
        applied.append("CLAHE")

    if brightness < 70:
        enhanced = gamma_correction(enhanced, gamma=1.6)
        applied.append("Gamma Correction (BRIGHT)")
    elif brightness > 180:
        enhanced = gamma_correction(enhanced, gamma=0.8)
        applied.append("Gamma Correction(DARK)")

    # If image is overall fine but slightly dull
    if len(applied) == 0:
        enhanced = gamma_correction(apply_clahe(enhanced), gamma=1.2)
        applied = ["CLAHE + Gamma Correction"]

    return enhanced, applied

# IMAGE PROCESSING

def process_image(path):
    img = cv.imread(path)
    if img is None:
        print("NOT ABLE TO READ IMAGE")
        return

    enhanced, applied = dynamic_enhance_logic(img)
    print(f"Applied: {', '.join(applied)}")

    cv.imshow(f"Enhanced ({', '.join(applied)})", enhanced)
    cv.waitKey(0)
    cv.destroyAllWindows()

# VIDEO PROCESSING

def process_video(path):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("nOT ABLE TO OPEN VIDEO")
        return

    save_output = True
    if save_output:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter('enhanced_dynamic_combo_output.mp4', fourcc,
                             int(cap.get(cv.CAP_PROP_FPS)),
                             (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    print("Dynamic multi-enhancement running... Press 'q' to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced, applied = dynamic_enhance_logic(frame)
        text = ", ".join(applied)
        cv.putText(enhanced, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Enhanced Video (Dynamic Multi)", enhanced)

        if save_output:
            out.write(enhanced)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:  # show updates every 30 frames
            print(f"Frame {frame_count}: {text}")

    cap.release()
    if save_output:
        out.release()
    cv.destroyAllWindows()



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
