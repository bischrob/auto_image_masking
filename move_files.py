import os
import shutil

src_folder = 'images'
dest_folder = 'PNGs'

# Ensure the destination folder exists
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Move each PNG file from the source to the destination folder
for file_name in os.listdir(src_folder):
    if file_name.lower().endswith('.png'):
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.move(src_path, dest_path)
        print(f"Moved {file_name} to {dest_folder}")

# Define the source and destination folders
source_folder = 'images'  # Replace with the path to your source folder
destination_folder = 'training'  # Replace with the path to your destination folder

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is an image and doesn't contain "_masked" in the filename
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and '_masked' not in filename:
        # Construct the full file paths
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)

        # Copy the image to the destination folder
        shutil.copy(src_path, dst_path)
        print(f"Copied: {filename}")
