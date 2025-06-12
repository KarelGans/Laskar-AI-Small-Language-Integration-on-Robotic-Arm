from ultralytics import YOLO
import torch



if __name__ == '__main__':
    # Load a YOLOv8 model (nano)
    print(torch.__version__)  # should show 2.1.2+cu121
    print(torch.cuda.is_available())  # should be True
    print(torch.cuda.get_device_name(0))  # should say RTX 4060 Ti

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # optional: cleans up inter-process memory
    model = YOLO('yolov8m.pt')

    # Train the model
    model.train(
        data='synthetic_dataset_2/data.yml',  # or data.yaml if you renamed it
        epochs=100,
        imgsz=1200,
        batch=4,
        device=0  # Use your first GPU
    )

    # Export the trained model to ONNX format
    model.export(format='onnx')

    # Validate and print metrics
    metrics = model.val()
    print(metrics)  # e.g. mAP, precision, recall
