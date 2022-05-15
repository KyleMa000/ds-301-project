import numpy as np # linear algebra
from torchvision import transforms
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from utils.model_utils import get_model

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

        label_color_map = [[0,0,255], [255,0,0],[0,255,0]]
        label_name_map = ['without', 'with', 'incorrect']
        label_id = int(annotation["labels"][idx].item()) - 1

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), label_color_map[label_id], 2)
        cv2.putText(
                img,
                label_name_map[label_id],
                (round(xmin), round(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color_map[label_id],
                2,
            )
        
    plt.imshow(img)
    plt.show()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

img = Image.open('./data/facemask_detection/images/maksssksksss1.png').convert("RGB")
img = np.asarray(img)

data_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

img = data_transform(img)
img = img.to(device)

model = get_model(4, 'mobilenetv3')
model.load_state_dict(torch.load('logs4/mobilenetv3/99.pth'))
model.eval()
model.to(device)

pred2 = model([img])

print("Predict with loaded model")
plot_image2(img, pred2[0])