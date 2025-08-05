# script to save the mask and bounding boxes from a segmentation model

import torch
import json
import numpy as np
from pycocotools import mask as mask_utils
# import cv2
from pathlib import Path


import json

def convert_to_label_studio_json(
    image_path: str,
    pred_boxes: list,
    image_width: int,
    image_height: int,
    label: str,
    model_version="one",
    prediction_score=0.5
):
    """
    Convert bounding boxes to Label Studio prediction format.

    Parameters:
        image_path (str): Path to the image as expected by Label Studio (e.g., "/static/img.jpg").
        pred_boxes (list of list): Each box is [x_min, y_min, x_max, y_max] in pixel units.
        image_width (int): Original width of the image in pixels.
        image_height (int): Original height of the image in pixels.
        labels (list of str): Label names corresponding to each bounding box. If None, defaults to "Unknown".
        model_version (str): Optional string identifying the model version.
        prediction_score (float): Optional overall prediction score.

    Returns:
        list: A list with one dictionary in Label Studio format.
    """
    results = []
    for i, box in enumerate(pred_boxes, start=1):

        x_min, y_min, x_max, y_max = box
        x_pct = (x_min / image_width) * 100
        y_pct = (y_min / image_height) * 100
        width_pct = ((x_max - x_min) / image_width) * 100
        height_pct = ((y_max - y_min) / image_height) * 100

        results.append({
            "id": f"result{i}",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "rotation": 0,
                "x": x_pct,
                "y": y_pct,
                "width": width_pct,
                "height": height_pct,
                "rectanglelabels": [label]
            }
        })

    return [{
        "data": {
            "image": image_path
        },
        "predictions": [{
            "model_version": model_version,
            "score": prediction_score,
            "result": results
        }]
    }]


# Example usage:
# convert_bboxes_to_ls(bboxes, "material/image.jpg", "outputs/ls_bboxes.json", image_width=2250, image_height=2250)

def extract_bounding_boxes(file_path: str):

    with open(file_path, "r") as f:
        data = json.load(f)

    bounding_boxes = []
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        bbox_converted = [x, y, x + w, y + h]
        bounding_boxes.append(bbox_converted)

    print(bounding_boxes)
    # Save to output file
    output_path=f"{file_path[:-5]}_bbox.json"
    with open(output_path, "w") as f:
        json.dump(bounding_boxes, f, indent=2)

    return bounding_boxes

# Example usage:
# bounding_boxes = extract_bounding_boxes(file_path="outputs/coco_annotations/result.json", output_path="outputs/coco_annotations/bboxes.txt")
