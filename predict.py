import numpy as np # linear algebra
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import cv2

def plot_image(img_tensor, annotation):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.detach().cpu()

    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    
    for box in annotation["boxes"]:
        box = box.detach().cpu().numpy()
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def plot_image2(img, annotation):
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()

    # workaround, convert to cv2 matrix
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, box in enumerate(annotation["boxes"]):
        box = box.detach().cpu().numpy()
        xmin, ymin, xmax, ymax = box

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
        cv2.putText(
                img,
                str(annotation["labels"][idx].item()),
                (round(xmin), round(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )
        
    plt.imshow(img)
    plt.show()


# imgs = list(sorted(os.listdir("data/facemask_detection/images/")))
# labels = list(sorted(os.listdir("data/facemask_detection/annotations/")))

# class MaskDataset(object):
#     def __init__(self, transforms):
#         self.transforms = transforms
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         self.imgs = list(sorted(os.listdir("data/facemask_detection/images/")))

#     def __getitem__(self, idx):
#         # load images ad masks
#         file_image = 'maksssksksss'+ str(idx) + '.png'
#         file_label = 'maksssksksss'+ str(idx) + '.xml'
#         img_path = os.path.join("data/facemask_detection/images/", file_image)
#         label_path = os.path.join("data/facemask_detection/annotations/", file_label)
#         img = Image.open(img_path).convert("RGB")
#         #Generate Label
#         target = generate_target(idx, label_path)
        
#         if self.transforms is not None:
#             img = self.transforms(img)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)

# data_transform = transforms.Compose([
#         transforms.ToTensor(), 
#     ])

# def collate_fn(batch):
#     return tuple(zip(*batch))

# dataset = MaskDataset(data_transform)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# for imgs, annotations in data_loader:
#         imgs = list(img.to(device) for img in imgs)
#         annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#         break

img = Image.open('./data/facemask_detection/archive/maksssksksss0.png').convert("RGB")
img = np.asarray(img)

data_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

img = data_transform(img)
img = img.to(device)

model2 = get_model_instance_segmentation(3)
model2.load_state_dict(torch.load('faster_rcnn.pth'))
model2.eval()
model2.to(device)

pred2 = model2([img])

print("Predict with loaded model")
plot_image2(img, pred2[0])