import torch
from utils.eval_utils import evaluate
from datasets.mask_dataset import MaskDataset, collate_fn
from utils.model_utils import get_model_instance_segmentation
import argparse

def main(mode, epoch):
    checkpoint_path = f'logs/{mode}/{epoch}.pth'

    dataset = MaskDataset(split='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(3, mode)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.to(device)

    evaluate(model, data_loader, device=device)


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
    args = parser.parse_args()

    mode = args.mode
    epoch = args.epoch
    main(mode, epoch)