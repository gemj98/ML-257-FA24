from ultralytics import YOLO
import torch

def train_yolov8():
    # Initialize YOLOv8 model
    # You can choose different model sizes: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO('yolov8n.pt')  # load pretrained model (recommended for training)
    
    # Train the model
    results = model.train(
        data='dataset/data.yaml',      # path to data.yaml file
        epochs=3,                    # number of epochs
        imgsz=640,                    # image size
        batch=16,                     # batch size
        workers=8,                    # number of worker threads
        device=0 if torch.cuda.is_available() else 'cpu',  # device to run on
        patience=50,                  # early stopping patience
        save=True,                    # save checkpoints
        project='runs/train',         # project name
        name='parking_lot_model',     # experiment name
        pretrained=True,             # use pretrained model
        optimizer='auto',            # optimizer (SGD, Adam, etc.)
        verbose=True,                # print verbose output
        seed=42,                     # random seed
        val=True,                    # validate during training
    )
    
    # Validate the model
    results = model.val()
    
    return model, results

if __name__ == "__main__":
    model, results = train_yolov8()
