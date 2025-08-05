# https://github.com/HumanSignal/label-studio-sdk/tree/master/src/label_studio_sdk/converter#coco

## examples
# Use the label-studio-converter to convert your COCO annotations to a format that Label Studio can understand:
label-studio-converter import coco -i /coco/dataset/annotations/instances_val2017.json -o ls-tasks.json --image-root-url "/data/local-files/?d=val2017"

label-studio-converter import coco -h
# now not needed direct coco format
## test
dir="/Users/avalos/Documents/Programming/image_segmentation_v2"
filename="11_52_06_Leman_Allaman_11-11-24-10_Li_Mo_crop_resized_crop"
label-studio-converter import coco \
  -i "$dir/outputs/${filename}.json" \
  -o "$dir/outputs/ls_${filename}.json" \
  --image-root-url "$dir/material/"
