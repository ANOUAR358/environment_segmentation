
import torch
import torch.nn as nn
from metrics import *
import json

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

from processing.data_preprocessing import SegmentationDataset


create_dataloaders=SegmentationDataset.create_dataloaders

model=None
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
##### use :scaler = torch.cuda.amp.GradScaler()

# Train with mixed precision
###########    use : with torch.cuda.amp.autocast():




# Loss Function with Class Weights
class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight  # Class weights
        
    def forward(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=1)
        
        dice_scores = []
        for class_idx in range(predictions.shape[1]):
            pred_class = predictions[:, class_idx]
            target_class = targets[:, class_idx]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            
            if self.weight is not None:
                dice_score *= self.weight[class_idx]
            
            dice_scores.append(dice_score)
            
        return 1 - torch.mean(torch.stack(dice_scores))

# Learning Rate Warmup
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




# Segmentation Trainer
class SegmentationTrainer:
    def __init__(self, model, device, num_classes=34, class_weights=None):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.criterion = MultiClassDiceLoss(weight=class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = get_lr_scheduler(self.optimizer, warmup_epochs=5, total_epochs=50)
        self.scaler = GradScaler()  # Mixed precision
        self.early_stopping = EarlyStopping(patience=10)  # Early stopping
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        total_correct = 0
        total_pixels = 0
        class_ious = np.zeros(self.num_classes)
        
        with tqdm(train_loader, desc='Training') as pbar:
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast():  # Mixed precision
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Calculate accuracy and IoU
                predictions = torch.argmax(outputs, dim=1)
                target_masks = torch.argmax(masks, dim=1)
                total_correct += (predictions == target_masks).sum().item()
                total_pixels += torch.numel(predictions)
                
                for class_idx in range(self.num_classes):
                    intersection = ((predictions == class_idx) & (target_masks == class_idx)).sum().float()
                    union = ((predictions == class_idx) | (target_masks == class_idx)).sum().float()
                    if union > 0:
                        class_ious[class_idx] += (intersection / union).cpu().numpy()
        
        accuracy = total_correct / total_pixels
        class_ious = class_ious / len(train_loader)
        mean_iou = np.mean(class_ious)
        
        return epoch_loss / len(train_loader), accuracy, mean_iou, class_ious
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        total_correct = 0
        total_pixels = 0
        class_ious = np.zeros(self.num_classes)
        
        with torch.no_grad():
            with tqdm(val_loader, desc='Validation') as pbar:
                for images, masks in pbar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    pbar.set_postfix({'val_loss': loss.item()})
                    
                    # Calculate accuracy and IoU
                    predictions = torch.argmax(outputs, dim=1)
                    target_masks = torch.argmax(masks, dim=1)
                    total_correct += (predictions == target_masks).sum().item()
                    total_pixels += torch.numel(predictions)
                    
                    for class_idx in range(self.num_classes):
                        intersection = ((predictions == class_idx) & (target_masks == class_idx)).sum().float()
                        union = ((predictions == class_idx) | (target_masks == class_idx)).sum().float()
                        if union > 0:
                            class_ious[class_idx] += (intersection / union).cpu().numpy()
        
        accuracy = total_correct / total_pixels
        class_ious = class_ious / len(val_loader)
        mean_iou = np.mean(class_ious)
        
        return val_loss / len(val_loader), accuracy, mean_iou, class_ious
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        best_miou = 0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_miou': [], 'val_miou': []}
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            train_loss, train_acc, train_miou, _ = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_miou, _ = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_miou'].append(train_miou)
            history['val_miou'].append(val_miou)
            
            print(f'Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
            print(f'Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}')
            print(f'Train mIoU: {train_miou:.4f} - Val mIoU: {val_miou:.4f}')
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_miou': val_miou,
                }, '/kaggle/working/best_model_loss.pth')
                print('Model saved (best loss)!')
            
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_miou': val_miou,
                }, '/kaggle/working/best_model_miou.pth')
                print('Model saved (best mIoU)!')
            
            if self.early_stopping(val_loss):
                print("Early stopping triggered!")
                break
        
        return history




# Plot Training History
def plot_training_history(history):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    ax3.plot(history['train_miou'], label='Train mIoU')
    ax3.plot(history['val_miou'], label='Val mIoU')
    ax3.set_title('Training and Validation mIoU')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mIoU')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_history.png')
    plt.close()

# Main Function
def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Define paths - Update these to your actual paths
        train_img_dir = 'data/train/images'  # Update this path
        train_mask_dir = 'data/train/masks'  # Update this path
        val_img_dir = 'data/val/images'      # Update this path
        val_mask_dir = 'data/val/masks'      # Update this path
        
        # Create model with 34 classes
        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b5',
            encoder_weights='imagenet',
            in_channels=3,
            classes=34,
        )
        model = model.to(device)
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_img_dir, train_mask_dir,
            val_img_dir, val_mask_dir,
            batch_size=4  # Reduced batch size for stability
        )
        
        # Initialize trainer
        class_weights = torch.tensor([1.0] * 34).to(device)  # Adjust weights if needed
        trainer = SegmentationTrainer(model, device, num_classes=34, class_weights=class_weights)
        
        # Train model
        history = trainer.train(train_loader, val_loader, num_epochs=50)
        
        # Plot training history
        plot_training_history(history)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

#if __name__ == '__main__':
#   main()


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, 
    num_epochs, device, checkpoint_path='/kaggle/working/model_checkpoint.pth'
):
    best_val_loss = float("inf")  # Initialize best metric to a very high value

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_ious = []
    val_ious = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        iou_list = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            targets = targets.squeeze(1)
            targets = targets.long()
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            accuracy = compute_accuracy(outputs, targets)
            running_corrects += accuracy * targets.size(0)
            total_samples += targets.size(0)

            # Compute IoU for the current batch
            ious = compute_mean_iou(outputs.argmax(dim=1), targets)
            iou_list.append(ious)
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects / total_samples
        avg_train_iou = torch.mean(torch.tensor(iou_list)).item()

        # Adjust learning rate
        scheduler.step()

        # Validation step
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        val_iou_list = []
        model.eval()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze(1)
                targets = targets.long()
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_corrects += compute_accuracy(outputs, targets) * targets.size(0)
                val_samples += targets.size(0)

                # Compute IoU for validation batch
                val_ious = compute_mean_iou(outputs.argmax(dim=1), targets)
                val_iou_list.append(val_ious)

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_accuracy = val_corrects / val_samples
        avg_val_iou = torch.mean(torch.tensor(val_iou_list)).item()

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Train IoU: {avg_train_iou:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
            f"Validation IoU: {avg_val_iou:.4f}"
        )
        if train_accuracy>val_accuracy+0.07 and epoch >10:
          print("the training process will be stoped : it's start overtfit to the validation , exiting...")
          #break

        # Store metrics at the end of each epoch
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_ious.append(avg_train_iou)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_ious.append(avg_val_iou)

    # After the training loop finishes, collect all metrics in a dictionary
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_ious': train_ious,
        'val_ious': val_ious,
    }

    # Convert the dictionary to JSON
    metrics_json = json.dumps(metrics)

    # Optionally save the metrics to a JSON file
    with open('training_metrics.json', 'w') as f:
        f.write(metrics_json)

    # Return the metrics dictionary
    return metrics
