


def compute_accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)  # Get the class with the highest probability
    targets = targets.long()  # Ensure targets are integers (long type)

    # Flatten the tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    correct_pixels = (predictions == targets).sum().item()
    total_pixels = targets.numel()  # Equivalent to batch_size * height * width
    accuracy = correct_pixels / total_pixels
    return accuracy


def compute_mean_iou(preds, targets, num_classes=30):
    """
    Compute the mean Intersection over Union (IoU) for the predictions.
    
    Args:
        preds (torch.Tensor): Predicted masks, shape (B, C, H, W).
        targets (torch.Tensor): Ground truth masks, shape (B, H, W).
        num_classes (int): Total number of classes in the segmentation task.
        
    Returns:
        float: Mean IoU of the predictions.
    """
    preds_flat = preds.argmax(dim=1).flatten()  # Convert to shape (B * H * W)
    targets_flat = targets.flatten()  # Flatten ground truth

    iou_scores = []
    
    for cls in range(num_classes):
        pred_cls = preds_flat == cls
        target_cls = targets_flat == cls
        
        intersection = torch.sum(pred_cls & target_cls).item()
        union = torch.sum(pred_cls | target_cls).item()
        
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    
    # Compute mean IoU, ignoring classes with no predictions and targets
    valid_iou_scores = [iou for iou in iou_scores if iou > 0]
    mean_iou = sum(valid_iou_scores) / len(valid_iou_scores) if valid_iou_scores else 0
    return mean_iou



def calculate_metrics(predictions, targets, num_classes=34, smooth=1e-7):
    """
    Calculate precision and recall for each class
    Args:
        predictions: Model predictions (after argmax)
        targets: Ground truth labels
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
    Returns:
        precisions: List of precision values for each class
        recalls: List of recall values for each class
    """
    precisions = []
    recalls = []
    
    # Calculate metrics for each class
    for class_idx in range(num_classes):
        # True Positives: Pixels correctly predicted as class_idx
        tp = ((predictions == class_idx) & (targets == class_idx)).sum().float()
        
        # False Positives: Pixels incorrectly predicted as class_idx
        fp = ((predictions == class_idx) & (targets != class_idx)).sum().float()
        
        # False Negatives: Pixels incorrectly predicted as not class_idx
        fn = ((predictions != class_idx) & (targets == class_idx)).sum().float()
        
        # Calculate precision and recall
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        
        precisions.append(precision.item())
        recalls.append(recall.item())
    
    # Calculate mean precision and recall
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    
    return precisions, recalls, mean_precision, mean_recall