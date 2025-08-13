import numpy as np
import json
import pycocotools.mask as mask_utils
from PIL import Image
import cv2


def sam_masks_to_coco(all_masks, image_path, json_outfile, image_id=1, category_id=1, category_name="mussel"):

    # Get image size
    image = Image.open(image_path)
    width, height = image.size

    # COCO format structure
    coco_output = {
        "images": [{
            "id": image_id,
            "file_name": image.filename.split("/")[-1],
            "width": width,
            "height": height
        }],
        "annotations": [],
        "categories": [{
            "id": category_id,
            "name": category_name
        }]
    }

    annotation_id = 1

    for i, mask in enumerate(all_masks):
        # Ensure mask is uint8 (required for RLE encoding)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Encode mask to RLE
        rle = mask_utils.encode(np.asfortranarray(mask))  # Fortran order required
        rle['counts'] = rle['counts'].decode('utf-8')  # COCO requires JSON-serializable strings

        # Compute area and bounding box
        area = float(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()  # [x, y, width, height]

        # Create COCO annotation
        annotation = {
            "id": annotation_id,
            "image_id": image_id,  # Must be defined elsewhere
            "category_id": category_id,  # Must be defined elsewhere
            "segmentation": rle,
            "area": area,
            "bbox": [float(x) for x in bbox],
            "iscrowd": 0
        }

        # Append annotation
        coco_output["annotations"].append(annotation)
        annotation_id += 1

    # Save to file
    with open(json_outfile, "w") as f:
        json.dump(coco_output, f)
    print("file saved to:", json_outfile)
