from ultralytics import YOLO
from torchvision import transforms
import numpy as np

# Dataset
import os, glob
from PIL import Image
from torch.utils.data import Dataset

class MaskedFaceTestDataset(Dataset):
    def __init__(self, root, transform=None):
        super(MaskedFaceTestDataset, self).__init__()
        self.imgs = sorted(glob.glob(os.path.join(root, '*.png')))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

def count_masks(test_dataset):
    """
    Counts the number of faces correctly wearing masks, without masks, and incorrectly wearing masks in each image.

    Args:
        test_dataset (MaskedFaceTestDataset): The test dataset containing images.
        model (YOLO): The trained YOLO model.

    Returns:
        np.ndarray: Array of shape (N, 3) with counts for [with_mask, without_mask, mask_weared_incorrect] per image.
    """
    model_path = 'runs/detect/train3/weights/best.pt'
    model = YOLO(model_path)

    num_images = len(test_dataset)
    counts = np.zeros((num_images, 3), dtype=np.int64)  # Columns: [with_mask, without_mask, mask_weared_incorrect]

    # Define class mapping as per YOLO training
    class_mapping = {
        'with_mask': 0,
        'without_mask': 1,
        'mask_weared_incorrect': 2
    }

    # Iterate through each image in the dataset with a progress bar
    for idx in range(num_images):
        img = test_dataset[idx]  # Get image tensor
        # Convert tensor back to PIL Image for YOLO inference
        img_pil = transforms.ToPILImage()(img)

        # Perform inference
        results = model(img_pil, imgsz=640)  # Model trained on imgsz 640

        # Initialise counters
        with_mask_count = 0
        without_mask_count = 0
        mask_weared_incorrect_count = 0

        # Parse detections
        # 'results' is a list; get the first (only) Result object
        result = results[0]

        # Iterate through each detected bounding box
        for box in result.boxes:
            class_id = int(box.cls.cpu().numpy()[0])  # Get class ID
            if class_id == class_mapping['with_mask']:
                with_mask_count += 1
            elif class_id == class_mapping['without_mask']:
                without_mask_count += 1
            elif class_id == class_mapping['mask_weared_incorrect']:
                mask_weared_incorrect_count += 1

        # Assign counts to the array
        counts[idx] = [with_mask_count, without_mask_count, mask_weared_incorrect_count]

    return counts

# Function to extract ground truth counts
def get_ground_truth_counts(labels_folder):
    """
    Extracts ground truth counts from YOLO-formatted label files.

    Args:
        labels_folder (str): Path to the labels directory (e.g., MaskedFace/train/labels).

    Returns:
        np.ndarray: Array of shape (N, 3) with counts for [with_mask, without_mask, mask_weared_incorrect].
    """
    label_files = sorted(glob.glob(os.path.join(labels_folder, '*.txt')))
    num_labels = len(label_files)
    counts = np.zeros((num_labels, 3), dtype=int)

    for idx, label_file in enumerate(label_files):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                class_id = int(parts[0])
                if class_id in [0, 1, 2]:
                    counts[idx][class_id] += 1

    return counts

# Function to compute MAPE
def compute_mape(true_counts, pred_counts):
    """
    Computes the Mean Absolute Percentage Error (MAPE) between true and predicted counts.

    Args:
        true_counts (np.ndarray): Ground truth counts, shape (N, 3).
        pred_counts (np.ndarray): Predicted counts, shape (N, 3).

    Returns:
        float: Average MAPE over all images and classes.
    """
    # Avoid division by zero by replacing zeros with ones in true_counts
    denominator = np.maximum(true_counts, 1)
    ape = np.abs((true_counts - pred_counts) / denominator) * 100
    mape = np.mean(ape)
    return mape

if __name__ == "__main__":
    # Define paths
    val_images_folder = "MaskedFace/val/images"
    val_labels_folder = "MaskedFace/val/labels"

    # Initialise the test dataset
    test_dataset = MaskedFaceTestDataset(
        root=val_images_folder,
        transform=transforms.ToTensor()
    )

    # Count masks
    mask_counts = count_masks(test_dataset)
    print("Mask Counts per Image:\n", mask_counts)

    # Extract ground truth counts
    ground_truth_counts = get_ground_truth_counts(val_labels_folder)
    print("Ground Truth Counts:\n", ground_truth_counts)

    # Calculate MAPE
    mape = compute_mape(ground_truth_counts, mask_counts)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
