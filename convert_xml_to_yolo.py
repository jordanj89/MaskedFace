import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image

# Define class labels and their corresponding IDs
class_mapping = {
    'with_mask': 0,
    'without_mask': 1,
    'mask_weared_incorrect': 2
}

def convert_xml_to_yolo(xml_file, images_folder, labels_folder, class_mapping):
    """
    Converts a single XML annotation file to YOLO format.

    Args:
        xml_file (str): Path to the XML file.
        images_folder (str): Path to the folder containing images.
        labels_folder (str): Path to the folder where YOLO .txt files will be saved.
        class_mapping (dict): Mapping from class names to class IDs.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract image filename
        filename = root.find('filename').text
        image_path = os.path.join(images_folder, filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping.")
            return

        # Open image to get dimensions
        image = Image.open(image_path)
        width, height = image.size

        yolo_annotations = []

        # Iterate through each object in the XML
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label not in class_mapping:
                print(f"Warning: Label '{label}' not recognized. Skipping object.")
                continue
            class_id = class_mapping[label]

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Ensure values are between 0 and 1
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            bbox_width = min(max(bbox_width, 0), 1)
            bbox_height = min(max(bbox_height, 0), 1)

            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # Define paths for the label file
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(labels_folder, txt_filename)

        # Write annotations to .txt file
        with open(txt_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')

    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

def create_labels_directories(base_dir, subsets=['train', 'val']):
    """
    Creates 'labels/train' and 'labels/val' directories if they don't exist.

    Args:
        base_dir (str): Path to the MaskedFace directory.
        subsets (list): List of subsets to process (default: ['train', 'val']).
    """
    for subset in subsets:
        labels_dir = os.path.join(base_dir, subset, 'labels')
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
            print(f"Created directory: {labels_dir}")

def convert_dataset_to_yolo(base_dir, class_mapping):
    """
    Converts the entire dataset from XML to YOLO format.

    Args:
        base_dir (str): Path to the MaskedFace directory.
        class_mapping (dict): Mapping from class names to class IDs.
    """
    subsets = ['train', 'val']
    for subset in subsets:
        print(f"\nProcessing {subset} subset...")
        subset_dir = os.path.join(base_dir, subset)
        images_folder = subset_dir  # Images are in the same folder
        labels_folder = os.path.join(subset_dir, 'labels')

        # Create labels directory if it doesn't exist
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)
            print(f"Created labels directory: {labels_folder}")

        # Find all XML files in the subset directory
        xml_files = glob.glob(os.path.join(subset_dir, '*.xml'))
        print(f"Found {len(xml_files)} XML files in {subset} subset.")

        # Iterate through all XML files and convert them
        for xml_file in xml_files:
            convert_xml_to_yolo(xml_file, images_folder, labels_folder, class_mapping)

    print("\nConversion to YOLO format completed successfully.")

if __name__ == "__main__":
    # Define the base directory where MaskedFace folder is located
    base_dir = 'MaskedFace/'  # Replace with your actual path

    # Ensure that the base_dir exists
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory '{base_dir}' does not exist. Please check the path.")

    # Convert the dataset to YOLO format
    convert_dataset_to_yolo(base_dir, class_mapping)
