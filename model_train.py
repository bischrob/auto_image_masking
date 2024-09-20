import os
import argparse
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torchvision import transforms
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.images = [f for f in os.listdir(image_dir) if not f.endswith('_masked.png')]  # Filter out masked images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.png', '_masked.png')  # Adjust extension if needed
        mask_path = os.path.join(self.mask_dir, mask_name) if self.mask_dir != self.image_dir else os.path.join(self.image_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else Image.new("L", image.size, 0)  # Handle missing masks

        # Ensure that the same transformations are applied to both image and mask
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize(self.image_size)(mask)  # Resize the mask
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor
            mask = (mask > 0.5).float()  # Ensure the mask is binary (0 or 1)

        return image, mask

def train_model(training_dir, model_path, mask_dir=None, num_epochs=15, log_file_path="training_log.txt", image_size=2048, batch_size=8):
    if mask_dir is None:
        print("Assuming that masks are stored as image_name{_masked}.{file_extension} as mask_dir is not specified.")
        mask_dir = training_dir

    # Define transformations (resize, convert to tensor, and normalize)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_dataset = ImageDataset(image_dir=training_dir,
                                         mask_dir=mask_dir,
                                         transform=train_transform,
                                         image_size=(image_size, image_size))

    dataset_size = len(image_dataset)
    print(f"Size of dataset is {str(dataset_size)} images.")
    
    # Load the pre-trained U-Net model
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the backbone
        encoder_weights="imagenet",     # Use weights pre-trained on ImageNet
        in_channels=3,                  # Input channels (3 for RGB)
        classes=1                       # Output classes (1 for binary segmentation)
    )

    # Loss function (Binary Cross-Entropy + Dice Loss can be used too)
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Open the log file to record the accuracy and epoch
    with open(log_file_path, "a") as log_file:
        for epoch in range(num_epochs):

            # Create a DataLoader for the dataset
            train_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Adjust num_workers for performance

            model.train()
            running_loss = 0.0

            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, masks)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Save the model with epoch number appended to the filename
            model_path_without_ext, _ = os.path.splitext(model_path)
            model_save_path = f"{model_path_without_ext}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved: {model_save_path}")

            # Log the accuracy (or loss) for each epoch
            log_file.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}\n")
            log_file.flush()  # Ensure it's written to the file immediately

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a U-Net model for image segmentation")
    parser.add_argument('--training_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with masks (if different from training_dir)')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--log_file_path', type=str, default="training_log.txt", help='Path to save training logs')
    parser.add_argument('--image_size', type=int, default=2048, help='Size to resize images and masks to')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')

    args = parser.parse_args()

    train_model(
        training_dir=args.training_dir,
        model_path=args.model_path,
        mask_dir=args.mask_dir,
        num_epochs=args.num_epochs,
        log_file_path=args.log_file_path,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
