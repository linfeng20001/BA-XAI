import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self):
        # Initializes the visualizer class.
        pass

    def tensor_to_pil(self, tensor):
        """
        Convert a PyTorch tensor to a PIL Image.
        """
        return Image.fromarray(tensor.byte().cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
    
    def show_images(self, original_image, ground_truth, prediction):
        """
        Displays the original image, ground truth, and prediction side-by-side.
        """
        # Convert tensors to PIL images if they are not already
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = self.tensor_to_pil(ground_truth)
        if isinstance(prediction, torch.Tensor):
            prediction = self.tensor_to_pil(prediction)
        
        # Create a matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original_image)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')

        axes[1].imshow(ground_truth)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(prediction)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
