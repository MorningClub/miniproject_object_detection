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
# This script is used to evaluate the model on the test data
#

# Define simple image transforms
train_transforms = T.Compose([T.ToTensor()])#, T.RandomHorizontalFlip(0.5)])
test_transforms = T.Compose([T.ToTensor()])

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

numbers = ["00", "01", "02", "03", "09", "10", "11", "12", "13", "14", "15", "17", "18"]

name = "my-dataset_test_network"
dataset_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/processed_to_images/Video00005_Test"

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    name=name,
)

# for i in numbers:

#     name = "my-dataset_test_network" + i
#     dataset_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/processed_to_images/Video000" + i + "_Ano"

#     # Create the dataset
#     dataset1 = fo.Dataset.from_dir(
#         dataset_dir=dataset_dir,
#         dataset_type=fo.types.COCODetectionDataset,
#         name=name,
#     )

#     dataset.merge_samples(dataset1)

# #create train and test view
# train_view = dataset.take(1100, seed=51)
# test_view = dataset.exclude([s.id for s in train_view])


torch_dataset = FiftyOneTorchDataset(dataset, transforms=test_transforms, classes=classes)



modelSetup = ModelSetup(9)
model = modelSetup.get_model()


#load model
checkpoint_file = os.path.join('saved_models', 'epic_model')
model.load_state_dict(torch.load(checkpoint_file))

torch_dataset.add_detections(
    model,
    torch_dataset,
    dataset,
    field_name="predictions"
)

#test againt high confidence predictions
high_conf_view = dataset.filter_labels("predictions", F("confidence") > 0.75)

results = fo.evaluate_detections(
    high_conf_view,
    "predictions",
    classes=None,
    eval_key="eval",
    compute_mAP=True
)

print(results.mAP())

plot_classes = [
    "car",
    "truck",
    "bus",
    "person",
    "rider"
]

results.print_report(classes=plot_classes)



plot = results.plot_pr_curves(classes=plot_classes)
plot.show()

#session = fo.launch_app(dataset)
#session.view = dataset.sort_by("eval_fp", reverse=True)



session = fo.launch_app(high_conf_view)
session.view = high_conf_view.sort_by("eval_fp", reverse=True)

session.wait()




# model.eval()
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# import time
# item = torch_dataset.__getitem__(0)[0]
# print(type(item))
# print(item.shape)
# print(item.dtype)
# item = item[None, :]
# print(item.shape)

# #model.to(device)
# item = item.cuda()

# start_time = time.time()
# res = model(item)
# print("seconds: ", time.time() - start_time)

# print(res)