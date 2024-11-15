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

def train_model(training_dir, model_path, mask_dir=None, test_dir=None, num_epochs=15, log_file_path="training_log.txt", image_size=2048, batch_size=8):
    if mask_dir is None:
        print("Assuming that masks are stored as image_name{_masked}.{file_extension} as mask_dir is not specified.")
        mask_dir = training_dir

    # Define transformations (resize, convert to tensor, and normalize)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally with a 50% probability
        transforms.RandomVerticalFlip(p=0.5),    # Randomly flip vertically with a 50% probability
        transforms.RandomRotation(degrees=15),   # Randomly rotate within a range of ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset and DataLoader
    image_dataset = ImageDataset(image_dir=training_dir,
                                         mask_dir=mask_dir,
                                         transform=train_transform,
                                         image_size=(image_size, image_size))
    train_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Optionally create a test dataset and DataLoader
    if test_dir:
        test_dataset = ImageDataset(image_dir=test_dir,
                                    mask_dir=test_dir,  # Assuming masks are in test_dir for simplicity
                                    transform=train_transform,
                                    image_size=(image_size, image_size))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, loss, and optimizer setup
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop with test evaluation
    with open(log_file_path, "a") as log_file:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
            log_file.write(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}\n")

            # Evaluation on test dataset if test_loader exists
            if test_dir:
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for images, masks in test_loader:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        loss = loss_fn(outputs, masks)
                        test_loss += loss.item()
                avg_test_loss = test_loss / len(test_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")
                log_file.write(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}\n")

            # Save model after each epoch
            model_path_without_ext, _ = os.path.splitext(model_path)
            model_save_path = f"{model_path_without_ext}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved: {model_save_path}")
            log_file.flush()  # Ensure it's written to the file immediately


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a U-Net model for image segmentation")
    parser.add_argument('--training_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory with masks (if different from training_dir)')
    parser.add_argument('--test_dir', type=str, help='Directory with test images')  # Add this line
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--log_file_path', type=str, default="training_log.txt", help='Path to save training logs')
    parser.add_argument('--image_size', type=int, default=2048, help='Size to resize images and masks to')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')

    args = parser.parse_args()

    train_model(
        training_dir=args.training_dir,
        model_path=args.model_path,
        mask_dir=args.mask_dir,
        test_dir=args.test_dir,  # Pass test_dir to the function
        num_epochs=args.num_epochs,
        log_file_path=args.log_file_path,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
