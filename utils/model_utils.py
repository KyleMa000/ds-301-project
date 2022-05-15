import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes, mode):
    # load an instance segmentation model pre-trained pre-trained on COCO
    if mode == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif mode == 'mobilenetv3':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    else:
        raise NotImplementedError('Mode {} is not implemented'.format(mode))

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model