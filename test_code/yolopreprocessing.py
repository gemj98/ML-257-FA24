import os
import shutil
from PIL import Image
import glob
from tqdm import tqdm

class DatasetPreprocessor:
    def __init__(self):
        # Define paths
        self.script_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.script_dir, 'data', 'PKLotYolov8')
        self.output_dir = os.path.join(self.script_dir, 'dataset')
        self.splits = ['train', 'valid', 'test']

    def create_directory_structure(self):
        """Create YOLO format directory structure"""
        for split in self.splits:
            os.makedirs(os.path.join(self.output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'labels', split), exist_ok=True)

    def process_dataset(self):
        """Process and organize the dataset into YOLO format"""
        self.create_directory_structure()

        for split in self.splits:
            # Define paths for current split
            images_src = os.path.join(self.data_dir, split, 'images')
            print(images_src)
            labels_src = os.path.join(self.data_dir, split, 'labels')
            print(labels_src)
            
            # Skip if split directory doesn't exist
            if not os.path.exists(images_src):
                continue

            # Get all image files
            image_files = glob.glob(os.path.join(images_src, '*.jpg')) + \
                         glob.glob(os.path.join(images_src, '*.png'))

            print(f"\nProcessing {split} split...")
            for img_path in tqdm(image_files):
                try:
                    # Get corresponding label path
                    img_name = os.path.basename(img_path)
                    base_name = os.path.splitext(img_name)[0]
                    label_path = os.path.join(labels_src, f"{base_name}.txt")

                    # Check if label file exists
                    if not os.path.exists(label_path):
                        continue

                    # Copy image to new location
                    shutil.copy2(
                        img_path, 
                        os.path.join(self.output_dir, 'images', split, img_name)
                    )

                    # Copy label file to new location
                    shutil.copy2(
                        label_path,
                        os.path.join(self.output_dir, 'labels', split, f"{base_name}.txt")
                    )

                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

    def create_data_yaml(self):
        """Create data.yaml file for YOLOv8"""
        yaml_content = f"""
path: {self.output_dir}
train: images/train
val: images/valid
test: images/test

nc: 2
names: ['empty', 'occupied']
        """
        
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content.strip())

def main():
    preprocessor = DatasetPreprocessor()
    
    print("Starting dataset preprocessing...")
    preprocessor.process_dataset()
    
    print("Creating data.yaml file...")
    preprocessor.create_data_yaml()
    
    print("Dataset preprocessing completed!")

if __name__ == "__main__":
    main()
