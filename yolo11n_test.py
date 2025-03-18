from ultralytics import YOLO
import cv2
import glob
import os

model = YOLO("yolo11s100E_best.onnx")  # Load the model
input_folder = "images/"  # Folder containing images
output_folder = "results/"  # Folder to save annotated images
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

# Process all images in the input folder
image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))  # Change to "*.png" if needed

for image_path in image_paths:
    results = model(image_path)  # Run inference
    
    for result in results:
        img = result.plot()  # Get annotated image with bounding boxes
        output_path = os.path.join(output_folder, os.path.basename(image_path))  # Save path
        cv2.imwrite(output_path, img)  # Save result

    print(f"Processed: {image_path}")

print("Batch processing completed!")