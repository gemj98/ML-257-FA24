from ultralytics import YOLO
import numpy as np

def validate_model(model_path, data_yaml):
    """
    Validate YOLOv8 model and print detailed metrics
    """
    # Load the model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        conf=0.25,          # confidence threshold
        iou=0.5,            # NMS IoU threshold
        verbose=True        # print detailed metrics
    )
    
    # Extract and print metrics
    print("\n=== Detailed Validation Metrics ===")
    print(f"mAP50      : {results.box.map50:.4f}")          # mAP at IoU=0.50
    print(f"mAP50-95   : {results.box.map:.4f}")           # mAP at IoU=0.50:0.95
    print(f"Precision  : {results.box.mp:.4f}")            # mean precision
    print(f"Recall     : {results.box.mr:.4f}")            # mean recall
    print(f"F1-Score   : {2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr):.4f}")
    
    # Print per-class metrics
    print("\n=== Per-Class Metrics ===")
    for i, class_name in enumerate(model.names.values()):
        print(f"\nClass: {class_name}")
        print(f"Precision  : {results.box.p[i]:.4f}")
        print(f"Recall     : {results.box.r[i]:.4f}")
        print(f"mAP50      : {results.box.ap50[i]:.4f}")
        print(f"mAP50-95   : {results.box.ap[i]:.4f}")

def main():
    # Set your paths
    model_path = '/home/amb/Desktop/ml257/parking-detection/test_code/runs/train/parking_lot_model2/weights/best.pt'
    data_yaml = 'dataset/data.yaml'
    
    print("Starting validation...")
    validate_model(model_path, data_yaml)

if __name__ == "__main__":
    main()
