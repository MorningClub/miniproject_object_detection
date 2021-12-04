import os
from torchvision import models
from modelSetup import ModelSetup
from fiftyoneDataset import FiftyOneTorchDataset
import fiftyone as fo
import detection.transforms as T 
import torch

#
#This script is used for training the model
#

#Transforms
train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
test_transforms = T.Compose([T.ToTensor()])

#Define possible classes
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

numbers = ["00", "01", "02", "03", "04", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]

name = "my-dataset"
dataset_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/processed_to_images/Video00000_Ano"

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    name=name,
)

for i in numbers:

    name = "my-dataset" + i
    dataset_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/processed_to_images/Video000" + i + "_Ano"

    # Create the dataset
    dataset1 = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.COCODetectionDataset,
        name=name,
    )

    dataset.merge_samples(dataset1)

#create train and test view
train_view = dataset.take(1600, seed=51)
test_view = dataset.exclude([s.id for s in train_view])

#create torch datasets
torch_dataset = FiftyOneTorchDataset(train_view, transforms=train_transforms, classes=classes)
torch_test_dataset = FiftyOneTorchDataset(test_view, transforms=test_transforms, classes=classes)

modelSetup = ModelSetup(9)
model = modelSetup.get_model()

modelSetup.do_training(model, torch_dataset, torch_test_dataset, num_epochs=15)

#save model
checkpoint_file = os.path.join('saved_models', 'epic_model')
torch.save(model.state_dict(), checkpoint_file)


torch_dataset.add_detections(
    model,
    torch_test_dataset,
    dataset,
    field_name="predictions"
)

results = fo.evaluate_detections(
    dataset,
    "predictions",
    classes=None,
    eval_key="eval",
    compute_mAP=True
)

print(results.mAP())

session = fo.launch_app(dataset)
session.view = dataset.sort_by("eval_fp", reverse=True)

session.wait()