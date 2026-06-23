from PIL import Image
from pathlib import Path

# ---------------------------------------------------------
# Crop coordinates (from ImageJ)
# ---------------------------------------------------------
def get_crop_coords(x, y, w, h, x_pixel=4056, y_pixel=3040):
    return (
        x / x_pixel,
        (x + w) / x_pixel,
        y / y_pixel,
        (y + h) / y_pixel
    )

# ---------------------------------------------------------
# Process dataset (NO augmentation here)
# ---------------------------------------------------------
def process_dataset(
    input_root=Path("Data_new_test"),
    output_root=Path("Data_clean"),
    target_size=(224, 224)
):
    x1, x2, y1, y2 = get_crop_coords(1740, 1032, 402, 762)

    splits = ["Train", "Validation", "Test"]

    for split in splits:
        for class_dir in (input_root / split).iterdir():
            if not class_dir.is_dir():
                continue

            input_path = class_dir
            output_path = output_root / split / class_dir.name
            output_path.mkdir(parents=True, exist_ok=True)

            for img_path in input_path.iterdir():
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                    continue

                img = Image.open(img_path).convert("RGB")

                # Crop
                w, h = img.size
                left   = int(x1 * w)
                right  = int(x2 * w)
                top    = int(y1 * h)
                bottom = int(y2 * h)

                cropped = img.crop((left, top, right, bottom))

                # Resize (force exact 224x224)
                resized = cropped.resize(target_size)

                # Save
                resized.save(output_path / img_path.name)

            print(f"Finished: {split}/{class_dir.name}")

if __name__ == "__main__":
    process_dataset(
        input_root=Path("data_masked"),
        output_root=Path("data_processed"),
        target_size=(224, 224)
    )
