import os
from PIL import Image


def crop_images_in_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # Process each file in the input folder and its subfolders
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                # Open image
                image_path = os.path.join(root, filename)
                img = Image.open(image_path)

                # Get image dimensions
                width, height = img.size

                # Calculate crop dimensions (10% from left)
                crop_width = int(width * 0.11)

                # Crop image (left, top, right, bottom)
                cropped_img = img.crop((crop_width, 0, width, height))
                # print(f"Cropping: Lose:{crop_width}, Actual width: {width}")

                # Determine relative path to output folder
                relative_path = os.path.relpath(root, input_folder)
                output_folder_path = os.path.join(output_folder, relative_path)

                # Create output subfolder if it doesn't exist
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)

                # Save cropped image
                output_path = os.path.join(output_folder_path, f'{filename}')
                cropped_img.save(output_path)

                # Close images
                img.close()
                cropped_img.close()


# Example usage
# input_folder = "./output_dir/2024-11-18/plates"
# output_folder = "./output_dir/2024-11-18/plates_new"
# crop_images_in_folder(input_folder, output_folder)
