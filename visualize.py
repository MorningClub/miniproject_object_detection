import fiftyone as fo

# dataset = fo.Dataset.from_dir(
#     #dataset_dir="C:/Users/alex_/OneDrive/Desktop/Skole/VisuellIntelligens/miniProject/Video00000_Ano", 
#     dataset_type=fo.types.COCODetectionDataset,
#     data_path="C:/Users/alex_/OneDrive/Desktop/Skole/VisuellIntelligens/miniProject/Video00000_Ano/data",
#     labels_path="C:/Users/alex_/OneDrive/Desktop/Skole/VisuellIntelligens/miniProject/Video00000_Ano"
# )

# counts = dataset.count_values("ground_truth.detections.label")

# print(counts)




name = "my-dataset"
dataset_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/processed_to_images/Video00003_Ano"

# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    name=name,
)

name = "my-dataset2"
dataset_dir = "C:/Users/alex_/Desktop/Skole/VisuellIntelligens/miniProject/processed_to_images/Video00000_Ano"

# Create the dataset
dataset2 = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    name=name,
)

dataset.merge_samples(dataset2)

# View summary info about the dataset
#print(dataset)

# Print the first few samples in the dataset
#print(dataset.head())

session = fo.launch_app(dataset)

session.wait()

counts = dataset.count_values("ground_truth.detections.label")

print(counts)