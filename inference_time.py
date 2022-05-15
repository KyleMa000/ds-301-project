# get inference time

import torch
from datasets.mask_dataset import MaskDataset, collate_fn
from utils.model_utils import get_model
import argparse
from tqdm import tqdm
from time import time

def main(mode, epoch, dataroot):
    checkpoint_path = f'logs3/{mode}/{epoch}.pth'

    dataset = MaskDataset(split='test', dataroot=dataroot)
    data_loader_test = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(4, mode)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.to(device)

    start_time = time()
    model.eval()
    with torch.no_grad():
        for imgs, annotations in tqdm(data_loader_test):
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            model(imgs, annotations)
    
    duration = time() - start_time
    print(f'Inference takes {duration} seconds. Performance {len(data_loader_test) / duration} FPS.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=None,
        type=str,
        help="[mobilenetv3 / resnet50]",
    )
    parser.add_argument(
        "--epoch",
        default=99,
        type=int,
        help="epoch num",
    )
    parser.add_argument(
        "--dataroot",
        default='./data/facemask_detection',
        type=str,
        help="path to the dataset root",
    )
    args = parser.parse_args()

    mode = args.mode
    epoch = args.epoch
    dataroot = args.dataroot

    main(mode, epoch, dataroot)