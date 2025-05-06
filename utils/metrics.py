# utils/metrics.py
import numpy as np
import torch

class ConfusionMatrix:
    """
    Computes a confusion matrix for evaluating semantic segmentation.

    Accumulates prediction and label masks over batches to compute
    Intersection over Union (IoU) and mean IoU (mIoU).
    """
    def __init__(self, num_classes, ignore_label=255):
        """
        Initializes the ConfusionMatrix.

        Args:
            num_classes (int): The number of classes in the dataset.
            ignore_label (int, optional): The label value to ignore in calculations. 
                                       Defaults to 255.
        """
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        # Initialize the confusion matrix as a numpy array of zeros
        # Dimensions: num_classes x num_classes
        # Rows: Ground Truth, Columns: Prediction
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds, labels):
        """
        Updates the confusion matrix with a new batch of predictions and labels.

        Args:
            preds (torch.Tensor or np.ndarray): Predicted segmentation masks. 
                                                Expected shape: (batch_size, height, width).
                                                Values should be class indices.
            labels (torch.Tensor or np.ndarray): Ground truth segmentation masks.
                                                 Expected shape: (batch_size, height, width).
                                                 Values should be class indices.
        """
        # Ensure preds and labels are numpy arrays on CPU
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            
        # Flatten arrays to easily process all pixels
        preds = preds.flatten()
        labels = labels.flatten()
        
        # Create a mask to filter out ignored labels and invalid label values
        mask = (labels != self.ignore_label) & (labels >= 0) & (labels < self.num_classes)
        
        # Apply the mask to get only valid labels and their corresponding predictions
        labels_valid = labels[mask]
        preds_valid = preds[mask]

        # Clip predictions to ensure they are within the valid class range [0, num_classes-1]
        # This handles potential invalid prediction values after masking.
        preds_valid = np.clip(preds_valid, 0, self.num_classes - 1)

        # Efficiently update confusion matrix using np.bincount
        # Create a unique index for each (label, prediction) pair
        indices = self.num_classes * labels_valid.astype(np.int64) + preds_valid
        # Count occurrences of each index
        counts = np.bincount(indices, minlength=self.num_classes**2)
        
        # Reshape the counts and add to the confusion matrix
        if len(counts) == self.num_classes**2: # Basic check for expected length
            self.matrix += counts.reshape(self.num_classes, self.num_classes)
        else:
            # This should ideally not happen if inputs are correct
            print(f"Warning: Bin count returned unexpected length: {len(counts)}. Confusion matrix might be inaccurate.")


    def compute_iou(self):
        """
        Computes the Mean Intersection over Union (mIoU) and per-class IoU 
        from the accumulated confusion matrix.

        Returns:
            tuple: A tuple containing:
                - mean_iou (float): The mean IoU score (average over valid classes).
                - iou_per_class (np.ndarray): An array containing the IoU for each class.
        """
        cm = self.matrix.astype(np.float64) # Use float64 for precision during division
        
        # Calculate intersection (diagonal elements)
        intersection = np.diag(cm)
        
        # Calculate union for each class
        # Union = Sum(Ground Truth) + Sum(Prediction) - Intersection
        ground_truth_set = cm.sum(axis=1) # Sum over columns (predictions) for each true label row
        predicted_set = cm.sum(axis=0)    # Sum over rows (true labels) for each predicted class column
        union = ground_truth_set + predicted_set - intersection

        # Compute IoU per class, handle division by zero (where union is 0)
        # np.divide handles division by zero, returning 0 where union is 0
        iou_per_class = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union!=0)
        
        # Calculate mean IoU only over classes that were present in the ground truth labels
        # This avoids penalizing for classes not present in the validation set
        present_classes_mask = (ground_truth_set > 0)
        
        # Filter IoU for present classes
        iou_present = iou_per_class[present_classes_mask]
        
        # Compute the mean, ignoring potential NaNs if a present class had 0 union (should be rare)
        mean_iou = np.nanmean(iou_present) if iou_present.size > 0 else 0.0
        
        # Return mIoU and per-class IoU as percentages
        return mean_iou * 100, iou_per_class * 100 

    def reset(self):
        """Resets the confusion matrix to all zeros."""
        self.matrix.fill(0)
