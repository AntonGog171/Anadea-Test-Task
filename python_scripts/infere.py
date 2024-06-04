import os
import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument('--model', type=str, choices=['big', 'small'], required=True, help='Model size keyword (big or small)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
    return parser.parse_args()


def load_model(model_keyword):
    if model_keyword == 'big':
        model_path = r"weights/big.pt"  # Update with the actual path to your big model weights
    elif model_keyword == 'small':
        model_path = r"weights/small.pt"  # Update with the actual path to your small model weights
    else:
        raise ValueError("Invalid model keyword. Choose 'big' or 'small'.")
    return YOLO(model_path)


def draw_predictions(image, masks):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for mask in masks:
        for segment in mask.xy:
            segment_np = torch.stack([torch.tensor(segment).float()]).cpu().numpy()
            segment_np = segment_np.reshape(-1, 1, 2).astype(int)
            cv2.polylines(image_rgb, [segment_np], isClosed=True, color=(0, 0, 255), thickness=2)
    return image_rgb

def run_inference(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_types=['dng', 'bmp', 'mpo', 'tif', 'jpeg', 'tiff', 'webp', 'pfm', 'png', 'jpg']
    image_names=list(filter(lambda x: True if any(ext in x for ext in file_types) else False, os.listdir(input_dir)))
    print(image_names)
    results = model.predict([os.path.join(input_dir,img_name) for img_name in image_names], iou=0.5, max_det=3000)
    for result, img_name in zip(results, image_names):
        image = cv2.imread(os.path.join(input_dir,img_name), cv2.IMREAD_GRAYSCALE)
        image_pred = draw_predictions(image, result.masks)
        cv2.imwrite(os.path.join(output_dir, img_name), image_pred) 
        
    print(f"Inference results saved to {output_dir}, original image size kept.")

def main():
    args = parse_args()
    model = load_model(args.model)
    run_inference(model, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
