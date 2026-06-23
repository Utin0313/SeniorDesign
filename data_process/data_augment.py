from PIL import Image
from pathlib import Path
import numpy as np

# ---------------------------------------------------------
# Add black noise to image
# ---------------------------------------------------------
def add_black_noise_to_image(img: Image.Image, noise_ratio=0.05):

    arr = np.array(img.convert("RGB"))

    h, w, _ = arr.shape
    total_pixels = h * w

    num_noisy = int(total_pixels * noise_ratio)

    ys = np.random.randint(0, h, num_noisy)
    xs = np.random.randint(0, w, num_noisy)

    arr[ys, xs] = [0, 0, 0]

    return Image.fromarray(arr)

# ---------------------------------------------------------
# Process dataset
# ---------------------------------------------------------
def process_dataset(
    input_root=Path("data_processed"),
    output_root=Path("data_final"),
    noise_ratio=0.05
):

    for split in ["Train", "Validation", "Test"]:

        split_input = input_root / split

        if not split_input.exists():
            print(f"Skipping: {split_input}")
            continue

        for class_folder in split_input.iterdir():

            if not class_folder.is_dir():
                continue

            out_class = output_root / split / class_folder.name
            out_class.mkdir(parents=True, exist_ok=True)

            for img_path in class_folder.iterdir():

                if img_path.suffix.lower() not in [
                    ".jpg", ".jpeg", ".png", ".bmp", ".tif"
                ]:
                    continue

                img = Image.open(img_path).convert("RGB")

                # -------------------------------------------------
                # TRAIN:
                # Save:
                #   1 clean original
                #   2 noisy augmented copies
                # -------------------------------------------------
                if split == "Train":

                    # Save clean image
                    img.save(out_class / img_path.name)

                    # Noisy copy 1
                    noisy_img1 = add_black_noise_to_image(
                        img,
                        noise_ratio
                    )

                    noisy_name1 = (
                        img_path.stem +
                        "_noise1" +
                        img_path.suffix
                    )

                    noisy_img1.save(out_class / noisy_name1)

                    # Noisy copy 2
                    noisy_img2 = add_black_noise_to_image(
                        img,
                        noise_ratio
                    )

                    noisy_name2 = (
                        img_path.stem +
                        "_noise2" +
                        img_path.suffix
                    )

                    noisy_img2.save(out_class / noisy_name2)

                # -------------------------------------------------
                # VALIDATION / TEST:
                # Save only clean image
                # -------------------------------------------------
                else:
                    img.save(out_class / img_path.name)

            print(f"Finished: {split}/{class_folder.name}")

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":

    process_dataset(
        input_root=Path("data_processed"),
        output_root=Path("data_final"),
        noise_ratio=0.05
    )
