# from tensorflow.keras.models import load_model
import cv2
import numpy as np

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
import base64
import io
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="https://facemask.dekun.me")
StringIO = io.StringIO


class objectDetection:
    def __init__(self):
        print("Initializing.....")
        self.data_transform = transforms.Compose([
            transforms.ToTensor(), 
        ])

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.get_model(4)
        self.model.load_state_dict(torch.load('mobilenetv3.pth'))
        self.model.eval()
        self.model.to(self.device)

        self.labels_dict = {0: "No Mask", 1: "Mask", 2: "Incorrect Mask"}  # dictionary to show the results
        self.color_dict = {
            0: (0, 0, 255),
            1: (0, 255, 0),
            2: (255, 0, 0)
        }  # color the box based on labels predicted green and red
        print("Initialization Finish")

    @staticmethod
    def get_model(num_classes, mode='mobilenetv3'):
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

    def annotate(self, img, annotation):
        for idx, box in enumerate(annotation[0]["boxes"]):
            box = box.detach().cpu().numpy()
            xmin, ymin, xmax, ymax = box

            class_num = annotation[0]['labels'][idx].item()
            if class_num >= 3:
                continue

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.color_dict[class_num], 2)
            cv2.putText(
                    img,
                    self.labels_dict[class_num],
                    (round(xmin), round(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    self.color_dict[class_num],
                    2,
                )

        return img


@app.route("/", methods=["POST", "GET"])
def homepage():
    return render_template("index.html")


@socketio.on("image")
def processImage(data_image):
    if str(data_image) == "data:,":
        pass
    else:
        sbuf = StringIO()
        sbuf.write(data_image)

        b = io.BytesIO(base64.b64decode(data_image))
        frame = Image.open(b).convert("RGB")
        img = Image.open(b).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        frame = od.data_transform(frame)
        frame = frame.to(od.device)
        annotation = od.model([frame])
        img = od.annotate(img, annotation)

        imgencode = cv2.imencode(".jpg", np.array(img))[1]
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = "data:image/jpg;base64,"
        stringData = b64_src + stringData

        emit("response_back", stringData)


if __name__ == "__main__":
    od = objectDetection()
    socketio.run(app, host='0.0.0.0', ssl_context=('fullchain.pem', 'privkey.pem'), port=5000)
