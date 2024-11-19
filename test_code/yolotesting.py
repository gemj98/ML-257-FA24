from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

class YOLOTester:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def test_single_image(self, image_path, conf_thresh=0.25):
        """Test model on a single image"""
        print(f"\nTesting image: {Path(image_path).name}")
        
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=conf_thresh,
            save=True,
            save_txt=True
        )[0]
        
        # Print detection results
        print("\nDetection Results:")
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = self.model.names[cls]
            print(f"Class: {class_name}, Confidence: {conf:.3f}")
            
        return results
    
    def test_folder(self, test_folder, conf_thresh=0.25):
        """Test model on all images in a folder"""
        test_files = list(Path(test_folder).glob('*.jpg')) + list(Path(test_folder).glob('*.png'))
        
        results_summary = {
            'total_images': len(test_files),
            'total_detections': 0,
            'class_counts': {},
            'confidence_scores': []  # Changed from avg_confidence to confidence_scores
        }
        
        print(f"\nTesting {len(test_files)} images...")
        
        for img_path in tqdm(test_files):
            results = self.model.predict(
                source=str(img_path),
                conf=conf_thresh,
                save=True
            )[0]
            
            # Collect statistics
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls]
                
                results_summary['total_detections'] += 1
                results_summary['confidence_scores'].append(conf)
                results_summary['class_counts'][class_name] = \
                    results_summary['class_counts'].get(class_name, 0) + 1
        
        # Calculate average confidence
        if results_summary['confidence_scores']:
            results_summary['avg_confidence'] = np.mean(results_summary['confidence_scores'])
        else:
            results_summary['avg_confidence'] = 0.0
        
        return results_summary

def print_test_summary(summary):
    """Print test results summary"""
    print("\n=== Test Results Summary ===")
    print(f"Total images processed: {summary['total_images']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")
    
    print("\nDetections per class:")
    for class_name, count in summary['class_counts'].items():
        print(f"{class_name}: {count}")
        print(f"Percentage: {(count/summary['total_detections']*100):.2f}%")

def main():
    # Set paths
    model_path = '/home/amb/Desktop/ml257/parking-detection/test_code/runs/train/parking_lot_model2/weights/best.pt'
    test_folder = '/home/amb/Desktop/ml257/parking-detection/test_code/data/PKLotYolov8/test/images'
    single_test_image = '/home/amb/Desktop/ml257/parking-detection/test_code/data/PKLotYolov8/test/images/2012-09-11_15_53_00_jpg.rf.8282544a640a23df05bd245a9210e663.jpg'  # replace with your test image
    
    # Initialize tester
    tester = YOLOTester(model_path)
    
    # 1. Test single image
    print("\n1. Testing single image...")
    results = tester.test_single_image(single_test_image)
    
    # 2. Test entire folder
    print("\n2. Testing folder...")
    summary = tester.test_folder(test_folder)
    print_test_summary(summary)

if __name__ == "__main__":
    main()


import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

class BoundingBoxVisualizer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.colors = {
            'empty': (0, 255, 0),     # Green for empty spots
            'occupied': (0, 0, 255)    # Red for occupied spots
        }
        
    def draw_boxes(self, image_path, conf_thresh=0.25):
        """Draw bounding boxes on image and return annotated image"""
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        results = self.model.predict(
            source=image_path,
            conf=conf_thresh,
            save=False
        )[0]
        
        # Draw boxes
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            conf = float(box.conf)
            
            # Get class name and color
            class_name = self.model.names[cls]
            color = self.colors['empty'] if class_name == 'empty' else self.colors['occupied']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f'{class_name} {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1-20), (x1+w, y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
        return image

    def visualize_predictions(self, image_path, save_path=None):
        """Visualize predictions with matplotlib"""
        annotated_image = self.draw_boxes(image_path)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
        
    def process_folder(self, folder_path, output_folder=None):
        """Process all images in a folder"""
        folder_path = Path(folder_path)
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(exist_ok=True)
            
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
        
        for img_path in image_files:
            print(f"Processing {img_path.name}")
            if output_folder:
                save_path = output_folder / f"annotated_{img_path.name}"
                self.visualize_predictions(img_path, save_path)
            else:
                self.visualize_predictions(img_path)

def main():
    # Initialize visualizer
    model_path = '/home/amb/Desktop/ml257/parking-detection/test_code/runs/train/parking_lot_model2/weights/best.pt'
    test_folder = '/home/amb/Desktop/ml257/parking-detection/test_code/data/PKLotYolov8/test/images'
    test_image = '/home/amb/Desktop/ml257/parking-detection/test_code/data/PKLotYolov8/test/images/2012-09-11_15_53_00_jpg.rf.8282544a640a23df05bd245a9210e663.jpg'  # replace with your test image
    
    visualizer = BoundingBoxVisualizer(model_path)
    
    # Process single image
    # test_image = 'dataset/images/test/test_image.jpg'  # replace with your test image
    visualizer.visualize_predictions(test_image)
    
    # Process folder
    output_folder = 'output/annotated_images'
    visualizer.process_folder(test_folder, output_folder)

if __name__ == "__main__":
    main()
