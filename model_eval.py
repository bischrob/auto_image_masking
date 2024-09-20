import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import argparse

def mask_images(img_dir, out_dir='masks', model_path='model.pth', probability=0.75, image_size=(256, 256)):
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the segmentation model with error handling
    try:
        model = smp.Unet(
            encoder_name="resnet34",        # Use ResNet-34 as the backbone
            encoder_weights=None,           # Do not use ImageNet weights
            in_channels=3,                  # Input channels (3 for RGB)
            classes=1                       # Output classes (1 for binary segmentation)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check if the image directory exists
    if not os.path.isdir(img_dir):
        print(f"Image directory not found: {img_dir}")
        return

    # Get a list of image files in the directory
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    img_paths = [
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.lower().endswith(supported_formats)
    ]

    if not img_paths:
        print(f"No image files found in directory: {img_dir}")
        return

    for img_path in img_paths:
        # Check if file exists before processing (redundant here but kept for consistency)
        if os.path.exists(img_path):
            print(f"Processing: {img_path}")
        else:
            print(f"File not found, skipping: {img_path}")
            continue  # Skip to the next image

        try:
            # Load the image
            img = Image.open(img_path).convert("RGB")
            original_width, original_height = img.size
            resized_width, resized_height = image_size

            # Preprocess the image
            preprocess = transforms.Compose([
                transforms.Resize((resized_height, resized_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(device)  # Move to device

            # Get the model prediction
            with torch.no_grad():
                output = model(input_tensor)

            # Process the output mask
            output = torch.sigmoid(output).squeeze().cpu().numpy()
            output_mask = (output > probability).astype(np.uint8)
            output_mask = 1 - output_mask

            # Resize the mask back to original size using LANCZOS
            mask_original_size = Image.fromarray((output_mask * 255).astype(np.uint8)).resize(
                (original_width, original_height), Image.Resampling.LANCZOS)

            # Save the mask
            os.makedirs(out_dir, exist_ok=True)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_filename = os.path.join(out_dir, img_name + '_mask.png')
            mask_original_size.save(mask_filename)

            print(f"Saved: {mask_filename}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue  # Skip to the next image

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict and save masks for images in a directory using a segmentation model.')
    parser.add_argument('img_dir', help='Path to the directory containing input images')
    parser.add_argument('--out_dir', default='masks', help='Directory to save the output masks')
    parser.add_argument('--model_path', default='model.pth', help='Path to the trained model weights')
    parser.add_argument('--probability', type=float, default=0.75, help='Probability threshold for mask prediction')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help='Image size to resize before prediction (width height)')
    args = parser.parse_args()

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found at {args.model_path}")
        return

    # Call the mask_images function
    mask_images(
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        model_path=args.model_path,
        probability=args.probability,
        image_size=tuple(args.image_size)
    )

if __name__ == '__main__':
    main()
