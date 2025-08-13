import os
from torch.nn import DataParallel
from models.geco_infer import build_model
from utils.arg_parser import get_argparser
import argparse
import torch
from torchvision import transforms as T
import matplotlib.patches as patches
from PIL import Image
from torchvision import ops
from utils.data import resize_and_pad
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from collections import OrderedDict
from extract_mask_and_bboxes import convert_to_label_studio_json, extract_bounding_boxes
import json
from utils.save_masks_to_coco import sam_masks_to_coco
import numpy as np
import cv2
import torch.nn.functional as F


bounding_boxes = []
global clicked

# Global variables to track drawing state
rect = None
start_x, start_y = None, None

# Event handler for mouse press (start drawing)
def on_press(event):
    global start_x, start_y, rect
    if event.inaxes:
        start_x, start_y = event.xdata, event.ydata  # Store starting point
        # Create a rectangle (but do not draw yet)
        rect = patches.Rectangle((start_x, start_y), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
        event.inaxes.add_patch(rect)
        plt.draw()  # Update plot to show rectangle (even if not yet drawn)

# Event handler for mouse motion (while drawing)
def on_motion(event):
    global start_x, start_y, rect
    if rect is not None and event.inaxes:
        # Update the width and height of the rectangle based on mouse position
        width = event.xdata - start_x
        height = event.ydata - start_y
        rect.set_width(width)
        rect.set_height(height)
        plt.draw()  # Redraw to update the rectangle while dragging

# Event handler for mouse release (end drawing)
def on_release(event):
    global rect
    # Once mouse is released, we finalize the bounding box
    if rect is not None:
        bounding_boxes.append([rect.get_x(), rect.get_y(), rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()])
        rect = None  # Reset rect after release
        print(bounding_boxes)


@torch.no_grad()
def demo(args, input_bboxes = False, bounding_boxes_json=None):
    global fig, ax
    global bounding_boxes

    print(args.model_path)

    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
        backends = "gpu"
    elif torch.backends.mps.is_available():
        print("CUDA is not available. Running on MPS.")
        backends = "mps"
    else:
        print("CUDA and MPS are not available. Running on CPU.")
        backends = "cpu"

    if backends == "gpu":
        gpu = 0
        torch.cuda.set_device(gpu)
        device = torch.device(gpu)
        model = DataParallel(
            build_model(args).to(device),
            device_ids=[gpu],
            output_device=gpu
        )
        model.load_state_dict(
            torch.load('GeCo.pth', weights_only=True)['model'], strict=False,
        )

        model.eval()

    if backends != "gpu":
        device = torch.device(backends)
        model = build_model(args).to(device)
        checkpoint = torch.load('MODEL_folder/GeCo.pth', map_location=device, weights_only=True)
        # Remove 'module.' from keys if present
        state_dict = checkpoint['model']
        new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(checkpoint['model'], strict=False)
        # strict=False allows loading models with missing or extra keys — useful if architectures changed slightly.
        model.eval()

        img_path = args.image_path
        image =  T.ToTensor()(Image.open(img_path).convert("RGB"))

    if input_bboxes == False:
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1,2,0))
        plt.axis('off')
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        plt.title("Click and drag to draw bboxes, then close window")
        plt.show()

        bboxes = torch.tensor(bounding_boxes, dtype=torch.float32)

    if input_bboxes == True:
        img_path = args.image_path
        image = T.ToTensor()(Image.open(img_path).convert("RGB"))
        # skip bounding box selection on image and load json
        bounding_boxes = extract_bounding_boxes(file_path=bounding_boxes_json)
        bounding_boxes = bounding_boxes[:10]
        bboxes = torch.tensor(bounding_boxes, dtype=torch.float32)


    img, bboxes, scale = resize_and_pad(image, bboxes, full_stretch=False)
    img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    outputs, _, _, _, masks = model(img, bboxes)
    del _
    idx = 0
    thr = 4
    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr],
                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr], 0.5)

    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr])[keep]
    
    bboxes = torch.clamp(boxes, 0, 1)

    plt.clf()
    plt.imshow(image.permute(1, 2, 0))

    if args.output_masks:
        # Get valid detection indices
        box_scores = outputs[idx]['box_v']
        threshold_mask = box_scores > (box_scores.max() / thr)

        if threshold_mask.dim() == 2:
            valid_detection_indices = torch.nonzero(threshold_mask, as_tuple=True)[1]
        else:
            valid_detection_indices = torch.nonzero(threshold_mask, as_tuple=True)[0]

        # define full_masks
        # Force conversion to tensor
        try:
            if isinstance(masks, list):
                full_mask = torch.stack(masks) if all(isinstance(m, torch.Tensor) for m in masks) else torch.tensor(
                    masks)
            else:
                full_mask = masks

            # Now try unique
            unique_ids = torch.unique(full_mask)
            print(f"Unique IDs: {unique_ids}")

        except Exception as e:
            print(f"Error processing masks: {e}")
            print(f"masks type: {type(masks)}")
            print(f"masks content preview: {str(masks)[:200]}...")


        print(f"Valid detection indices: {valid_detection_indices}")
        print(f"Available masks: {full_mask[0].shape[0]}")  # Should be 449

        # Extract individual masks
        all_masks_for_coco = []
        masks_tensor = full_mask[0]  # Shape: [449, 1024, 1024]

        for i, detection_idx in enumerate(valid_detection_indices):
            detection_idx_val = detection_idx.item()

            # Check if detection index is within bounds
            if detection_idx_val < masks_tensor.shape[0]:
                # Get the individual mask
                individual_mask = masks_tensor[detection_idx_val]  # Shape: [1024, 1024]

                # Convert to binary mask (boolean to uint8)
                binary_mask = individual_mask.cpu().numpy().astype(np.uint8)

                # Check if mask has content
                if binary_mask.sum() > 0:
                    # print(f"Detection {i}: Mask {detection_idx_val} has {binary_mask.sum()} pixels")

                    # Resize if needed to match original image size
                    if binary_mask.shape != (image.shape[1], image.shape[2]):
                        target_h, target_w = image.shape[1], image.shape[2]
                        binary_mask_resized = cv2.resize(
                            binary_mask.astype(np.float32),
                            (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)
                        all_masks_for_coco.append(binary_mask_resized)
                    else:
                        all_masks_for_coco.append(binary_mask)
                else:
                    print(f"Detection {i}: Mask {detection_idx_val} is empty")
            else:
                print(f"Detection {i}: Index {detection_idx_val} out of bounds (max: {masks_tensor.shape[0] - 1})")

        print(f"Total valid masks for COCO: {len(all_masks_for_coco)}")

        # Create visualization by combining the selected masks
        N_masks = len(all_masks_for_coco)
        if N_masks > 0:

            target_h = int(img.shape[2] / scale)
            target_w = int(img.shape[3] / scale)
            combined_mask = torch.zeros(target_h, target_w, dtype=torch.int)

            for i, detection_idx in enumerate(valid_detection_indices[:len(all_masks_for_coco)]):
                if detection_idx.item() < masks_tensor.shape[0]:
                    mask_small = masks_tensor[detection_idx.item()]  # [H_mask, W_mask]
                    mask_resized = T.Resize(
                        (target_h, target_w),
                        interpolation=T.InterpolationMode.NEAREST
                    )(mask_small.unsqueeze(0)).squeeze(0).bool()
                    combined_mask[mask_resized] = i + 1

            mask_display = combined_mask[:image.shape[1], :image.shape[2]]

            # Overlay
            cmap = plt.cm.tab20
            norm = plt.Normalize(vmin=0, vmax=N_masks)
            print(f"Number of masks to display: {N_masks}")
            # Create a shuffled list of IDs from 1..N_masks
            perm = np.random.permutation(np.arange(1, N_masks + 1))

            # Map original IDs to shuffled ones
            mask_ids = mask_display.clone()
            for old_id, new_id in zip(range(1, N_masks + 1), perm):
                mask_ids[mask_display == old_id] = new_id

            # Apply colors to shuffled IDs
            rgba_image = cmap(norm(mask_ids))
            rgba_image[mask_display == 0, -1] = 0
            plt.imshow(rgba_image, alpha=0.6)

            # Prepare masks for COCO saving
            separate_masks = [(mask_display == mask_id).numpy().astype(np.uint8)
                              for mask_id in range(1, N_masks + 1)]

            # Save to COCO
            sam_masks_to_coco(
                all_masks=separate_masks,
                image_path=args.image_path,
                json_outfile=args.json_outfile,
                image_id=idx + 1,
                category_id=1,
                category_name="mussel"
            )

    pred_boxes = bboxes.cpu() / torch.tensor([scale, scale, scale, scale]) * img.shape[-1]
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]

        plt.plot([box[0], box[0], box[2], box[2], box[0]], [box[1], box[3], box[3], box[1], box[1]], linewidth=0.7,
                 color='orange')

    pred_boxes = bounding_boxes
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        plt.plot([box[0], box[0], box[2], box[2], box[0]], [box[1], box[3], box[3], box[1], box[1]], linewidth=2,
                 color='red')
    plt.title("Number of selected objects:" + str(len(bboxes)))
    plt.axis('off')

    # Save figure
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(args.save_path, bbox_inches='tight', dpi=350)
    print(f"Saved figure to: {args.save_path}")
    plt.show()

#     pred_boxes = bboxes.cpu() / torch.tensor([scale, scale, scale, scale]) * img.shape[-1]

    # Save prediction boxes
    # json_output = convert_to_label_studio_json(
    #     image_path=args.image_path,
    #     pred_boxes=pred_boxes.tolist(),
    #     image_width=image.shape[1],
    #     image_height=image.shape[2],
    #     label= "Mussel"
    # )

    # Save to file
    # with open("outputs/label_studio_predictions.json", "w") as f:
    #     json.dump(json_output, f, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeCo', parents=[get_argparser()])
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    args.image_path = "../DATA_Rovailake/material/Photo15-07-13_Leman_SaintPrex_19-04-2025_20_Li_Mo_crop_resized_crop.jpg"
    args.filename = os.path.basename(args.image_path.strip())
    args.json_outfile = f"../DATA_Rovailake/instance_segmentation/GeCO/{args.filename[:-4]}_coco_masks.json"

    base_path = f"../DATA_Rovailake/instance_segmentation/GeCO/{args.filename[:-4]}_masks"
    save_path = f"{base_path}.jpg"
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base_path}_{counter}.jpg"
        counter += 1
    print(save_path)
    args.save_path = save_path
    print(args.output_masks)
    demo(args)
