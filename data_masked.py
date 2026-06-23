import os
import cv2
import glob
import numpy as np


def generate_brightness_mask(image_path, brightness_min, brightness_max,
                              dot_saturation_min=80):
    # Read image from file
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Original brightness mask (same logic as before) ---
    brightness = hsv[:, :, 2]           # V channel
    bg_mask = cv2.inRange(brightness,   # pixels TO REMOVE (background)
                          brightness_min,
                          brightness_max)

    # -- Colored dot rescue mask --
    saturation = hsv[:, :, 1]           # S channel
    dot_mask = (saturation >= dot_saturation_min).astype(np.uint8) * 255

    # -- Combine: remove background UNLESS it is a colored dot --
    remove_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(dot_mask))

    # Keep mask = inverse of remove mask
    keep_mask = cv2.bitwise_not(remove_mask)

    # Apply to image
    keep_mask_3 = keep_mask[:, :, None]           # expand to 3 channels
    result = (img * (keep_mask_3 / 255)).astype(np.uint8)

    return result


# ---------------------------------------------------------
# Function: process_dataset
# ---------------------------------------------------------
def process_dataset(
    data_root="Data",
    out_root="Data_Test",
    brightness_min=0,
    brightness_max=120,
    dot_saturation_min=80,         
    splits=("Train", "Validation", "Test"),
    exts=("jpg", "jpeg", "png")
):
    for split in splits:
        split_input_path = os.path.join(data_root, split)
        if not os.path.isdir(split_input_path):
            print(f"Skipping missing folder: {split_input_path}")
            continue

        for class_name in os.listdir(split_input_path):
            class_input_path = os.path.join(split_input_path, class_name)
            if not os.path.isdir(class_input_path):
                continue

            class_output_path = os.path.join(out_root, split, class_name)
            os.makedirs(class_output_path, exist_ok=True)

            image_paths = []
            for ext in exts:
                image_paths.extend(
                    glob.glob(os.path.join(class_input_path, f"*.{ext}"))
                )

            if not image_paths:
                print(f"No images found in: {class_input_path}")
                continue

            for path in image_paths:
                masked_image = generate_brightness_mask(
                    path,
                    brightness_min,
                    brightness_max,
                    dot_saturation_min
                )
                base_name = os.path.splitext(os.path.basename(path))[0]
                output_path = os.path.join(
                    class_output_path, f"{base_name}_masked.jpg"
                )
                cv2.imwrite(output_path, masked_image)

            print(f"Finished: {split}/{class_name} ({len(image_paths)} images)")


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    process_dataset(
        data_root="Data",
        out_root="data_masked",
        brightness_min=0,
        brightness_max=225,
        dot_saturation_min=80       
    )
