import os
from PIL import Image

def convert_rgba_to_bw_mask(input_folder, output_folder):
    """
    Convert RGBA images in the input_folder to black and white masks in JPEG format.
    The alpha channel is used to create the mask (alpha=0 becomes black, alpha>0 becomes white).
    
    Args:
    - input_folder (str): Path to the folder containing RGBA PNG images.
    - output_folder (str): Path to the folder where JPEG masks will be saved.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.png'):
            # Open the image
            img_path = os.path.join(input_folder, file_name)
            img = Image.open(img_path)
            
            # Check if the image is RGBA
            if img.mode == 'RGBA':
                # Extract the alpha channel
                alpha = img.getchannel('A')
                
                # Convert the alpha channel to a black and white mask
                # Alpha values greater than 0 become white, 0 values become black
                mask = alpha.point(lambda p: 0 if p > 0 else 255)
                
                # Save the result as JPEG
                output_file_name = os.path.splitext(file_name)[0] + '.jpg'
                output_path = os.path.join(output_folder, output_file_name)
                mask.convert('RGB').save(output_path, 'JPEG')
                print(f"Converted {file_name} to {output_file_name}")
            else:
                print(f"Skipping non-RGBA image: {file_name}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert RGBA PNG images to black and white masks in JPEG format.")
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing RGBA PNG images.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save JPEG masks.')
    
    args = parser.parse_args()
    
    convert_rgba_to_bw_mask(args.input_folder, args.output_folder)



# import os
# import shutil

# src_folder = 'images'
# dest_folder = 'PNGs'

# # Ensure the destination folder exists
# if not os.path.exists(dest_folder):
#     os.makedirs(dest_folder)

# # Move each PNG file from the source to the destination folder
# for file_name in os.listdir(src_folder):
#     if file_name.lower().endswith('.png'):
#         src_path = os.path.join(src_folder, file_name)
#         dest_path = os.path.join(dest_folder, file_name)
#         shutil.move(src_path, dest_path)
#         print(f"Moved {file_name} to {dest_folder}")
