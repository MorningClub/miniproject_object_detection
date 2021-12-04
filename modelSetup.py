import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection.engine import train_one_epoch, evaluate
import detection.utils as utils

class ModelSetup:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_model(self): #basert på post i teams

        #load pretrained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        #get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        #replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model

    def do_training(self, model, torch_dataset, torch_dataset_test, num_epochs=4):
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            torch_dataset, batch_size=2, shuffle=True,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            torch_dataset_test, batch_size=1, shuffle=False,
            collate_fn=utils.collate_fn)

        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using device %s" % device)

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=3,
                                                        gamma=0.1)

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)