import os
from torchvision import models
from modelSetup import ModelSetup
from fiftyoneDataset import FiftyOneTorchDataset
import fiftyone as fo
import detection.transforms as T 
import torch
import numpy as np
from fiftyone import ViewField as F

#
#This script is used to draw labels for the test video 
#

#create transforms
test_transforms = T.Compose([T.ToTensor()])

#load model
modelSetup = ModelSetup(9)
model = modelSetup.get_model()

checkpoint_file = os.path.join('saved_models', 'epic_model')
model.load_state_dict(torch.load(checkpoint_file))

model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define possible classes
classes = [
    "background",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "scooter",
    "person",
    "rider"
]

#generate dataset
name = "my-dataset"
dataset_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/processed_to_images/Video00005_Test"
output_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/output_images"

# dataset = fo.Dataset.from_images_dir(dataset_dir)
# dataset.evaluate_detections(
#     "predictions", gt_field="ground_truth", eval_key="eval"
# )

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    name=name,
)

torch_dataset = FiftyOneTorchDataset(dataset, transforms=test_transforms, classes=classes)

torch_dataset.add_detections(
    model,
    torch_dataset,
    dataset,
    field_name="predictions"
)

#Render with labels
label_fields = ["predictions"]
high_conf_view = dataset.filter_labels("predictions", F("confidence") > 0.75)

high_conf_view.draw_labels(output_dir, label_fields=label_fields)

session = fo.launch_app(dataset)
#session.view = dataset.sort_by("eval_fp", reverse=True)

session.wait()


# item = torch_dataset.__getitem__(0)[0]
# print(type(item))
# print(item.shape)
# print(item.dtype)
# item = item[None, :]
# print(item.shape)

# # model.to(device)
# # item.to(device)
# res = model(item)
# print(res)