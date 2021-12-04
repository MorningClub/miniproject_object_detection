import torch
import fiftyone.utils.coco as fouc
from PIL import Image
import fiftyone as fo

class FiftyOneTorchDataset(torch.utils.data.Dataset):

    def __init__(self, fiftyone_dataset, transforms=None, gt_field="ground_truth", classes=None):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field
        self.img_paths = self.samples.values("filepath")
        self.classes = classes

        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det, metadata=metadata, category_id=category_id
            )

            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

    def convert_torch_predictions(self, preds, det_id, s_id, w, h, classes):
        # Convert the outputs of the torch model into a FiftyOne Detections object
        dets = []
        for bbox, label, score in zip(
            preds["boxes"].cpu().detach().numpy(), 
            preds["labels"].cpu().detach().numpy(), 
            preds["scores"].cpu().detach().numpy()
        ):
            # Parse prediction into FiftyOne Detection object
            x0,y0,x1,y1 = bbox
            coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1-x0, y1-y0])
            det = coco_obj.to_detection((w,h), classes)
            det["confidence"] = float(score)
            dets.append(det)
            det_id += 1
            
        detections = fo.Detections(detections=dets)
            
        return detections, det_id

    def add_detections(self, model, torch_dataset, view, field_name="predictions"):
        # Run inference on a dataset and add results to FiftyOne
        torch.set_num_threads(1)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using device %s" % device)

        model.eval()
        model.to(device)
        image_paths = torch_dataset.img_paths
        classes = torch_dataset.classes
        det_id = 0
        
        with fo.ProgressBar() as pb:
            for img, targets in pb(torch_dataset):
                # Get FiftyOne sample indexed by unique image filepath
                img_id = int(targets["image_id"][0])
                img_path = image_paths[img_id]
                sample = view[img_path]
                s_id = sample.id
                w = sample.metadata["width"]
                h = sample.metadata["height"]
                
                # Inference
                preds = model(img.unsqueeze(0).to(device))[0]
                
                detections, det_id = self.convert_torch_predictions(
                    preds, 
                    det_id, 
                    s_id, 
                    w, 
                    h, 
                    classes,
                )
                
                sample[field_name] = detections
                sample.save()