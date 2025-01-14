import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SegmentationPredictor:
    def __init__(self, model, input_size=(128, 128), device="cpu"):
        """
        Initialize the SegmentationPredictor class.
        
        Args:
            model (torch.nn.Module): Trained DeepLabV3+ model.
            input_size (tuple): Input size for the model (height, width).
            device (str): Device to use for computation ("cuda" or "cpu").
        """
        self.model = model
        self.input_size = input_size
        self.device = device

    def preprocess_image(self, image):
        """
        Preprocess the input image to match the model requirements.
        
        Args:
            image (PIL.Image or np.array): Input image.
            
        Returns:
            torch.Tensor: Preprocessed image tensor with shape (1, 3, height, width).
        """
        transform = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ImageNet
        ])
        return transform(image).unsqueeze(0)

    def predict_mask(self, image_path):
        """
        Perform segmentation prediction using a DeepLabV3+ model.
        
        Args:
            image_path (str): Path to the input image.
            
        Returns:
            np.array: Predicted segmentation mask with shape (height, width).
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Perform the prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.argmax(output, dim=1)  # Get the predicted segmentation mask
        
        # Convert the output tensor to a NumPy array
        predicted_mask = output.cpu().squeeze().numpy()
        return predicted_mask

    def predict(self, dataset):
        """
        Perform predictions for a dataset.
        
        Args:
            dataset (list of str): List of image paths.
            
        Returns:
            list of np.array: List of predicted segmentation masks.
        """
        all_preds = []
        
        for image_path in dataset:
            # Predict the segmentation mask
            pred_mask = self.predict_mask(image_path)
            
            # Append predictions to the list
            all_preds.append(pred_mask)
        
        return all_preds

    def visualize_predictions(self, dataset, num_images=5):
        """
        Visualize the predictions for a number of images.
        
        Args:
            dataset (list of str): List of image paths.
            num_images (int): Number of images to visualize.
        """
        # Ensure num_images does not exceed the dataset size
        num_images = min(num_images, len(dataset))
        
        plt.figure(figsize=(15, num_images * 5))
        
        for i, image_path in enumerate(dataset[:num_images]):
            # Predict the segmentation mask
            pred_mask = self.predict_mask(image_path)
            
            # Load the original image
            image = Image.open(image_path).convert("RGB")
            
            # Plot the original image and the predicted mask
            plt.subplot(num_images, 2, 2 * i + 1)
            plt.imshow(image)
            plt.title(f"Original Image {i+1}")
            plt.axis("off")
            
            plt.subplot(num_images, 2, 2 * i + 2)
            plt.imshow(image)
            plt.imshow(pred_mask, cmap="jet", alpha=0.5)  # Overlay the predicted mask with transparency
            plt.title(f"Predicted Mask {i+1}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
