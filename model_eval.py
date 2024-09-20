import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import numpy as np
import os
from PIL import Image, ImageChops, ImageOps
from scipy.ndimage import label, find_objects
import glob
import re
import shutil

def crop_image(img_path, out_path, model_path, num_objects = 1, probability=0.75):

    # Check if file exists before attempting to delete it
    if os.path.exists(img_path):
        print(f"Processing: {img_path}")
    else:
        print(f"File not found, skipping: {img_path}")
        return None
   
    # Load the segmentation model
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the backbone
        encoder_weights=None,           # Do not use ImageNet weights, as we're loading fine-tuned weights
        in_channels=3,                  # Input channels (3 for RGB)
        classes=1                       # Output classes (1 for binary segmentation)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load the image
    img = Image.open(img_path).convert("RGB")
    original_width, original_height = img.size
    resized_width, resized_height = 2048, 2048
    
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((resized_width, resized_height)), 
        transforms.ToTensor(),        
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Get the model prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process the output mask
    output = torch.sigmoid(output).squeeze().cpu().numpy()
    output_mask = (output > probability).astype(np.uint8)
    
    # Create directories for saving cropped images
    
    os.makedirs(out_path, exist_ok=True)

    # Resize the original image and mask using LANCZOS interpolation
    img_resized = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    mask_resized = Image.fromarray((output_mask * 255).astype(np.uint8)).resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    # Label and find the objects in the mask
    labeled_mask, num_features = label(np.array(mask_resized))
    print(f"Number of objects found: {num_features}")
    
    # Calculate scaling factors for coordinates
    x_scale = original_width / resized_width
    y_scale = original_height / resized_height
    objects = find_objects(labeled_mask)

    # Resize the mask back to original size using LANCZOS
    mask_original_size = Image.fromarray((output_mask * 255).astype(np.uint8)).resize((original_width, original_height), Image.Resampling.LANCZOS)

    # Create an alpha channel based on the resized mask
    mask_array = np.array(mask_original_size)
    alpha_channel = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    alpha_image = Image.fromarray(alpha_channel, mode='L')

    # Add alpha channel to the original image
    img_rgba = img.convert("RGBA")
    img_with_alpha = ImageChops.multiply(img_rgba, Image.merge("RGBA", [img_rgba.split()[0], 
                                                                        img_rgba.split()[1], 
                                                                        img_rgba.split()[2], 
                                                                        alpha_image]))


    # Iterate through detected objects and crop them
    for i, obj_slice in enumerate(objects[:num_objects]):
        left = max(0, int(obj_slice[1].start * x_scale) - border_size)
        upper = max(0, int(obj_slice[0].start * y_scale) - border_size)
        right = min(original_width, int(obj_slice[1].stop * x_scale) + border_size)
        lower = min(original_height, int(obj_slice[0].stop * y_scale) + border_size)
        
        # Crop the object with the alpha channel and add a border
        cropped_img = img_with_alpha.crop((left, upper, right, lower))
        cropped_img_with_border = ImageOps.expand(cropped_img, border=border_size, fill=(0, 0, 0, 0))
        
        # Save the cropped image
        img_base_name = os.path.splitext(os.path.basename(img_path))[0]
        img_name = os.path.join(out_path, f"{img_base_name}_side-{i+1}.png")
        cropped_img_with_border.save(img_name, format='PNG')
        print(f"Saved: {img_name}")
