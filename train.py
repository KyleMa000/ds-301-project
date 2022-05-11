from tqdm import tqdm
from utils.model_utils import get_model_instance_segmentation
import torch
from datasets.mask_dataset import MaskDataset, collate_fn
from time import time
import os
import argparse

def main(mode, logpath):
    num_epochs = 100

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(3, mode=mode)
    model.to(device)

    dataset = MaskDataset(split='train')
    dataset_test = MaskDataset(split='test')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, collate_fn=collate_fn)
        
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(data_loader) - 1)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    len_dataloader = len(data_loader)

    print(f'Training {mode}...')
    output_dir = f'{logpath}/{mode}'
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'log.txt')
    log_file = open(log_file, 'w')
    start_time = time()


    for epoch in range(num_epochs):
        model.train()
        i = 0    
        epoch_loss = 0
        tqdm_bar = tqdm(data_loader, desc="Epoch {}/{}".format(epoch+1, num_epochs))
        for imgs, annotations in tqdm_bar:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
            tqdm_bar.set_description(f'Iteration: {i}/{len_dataloader}, Loss: {losses: .2f}')
            epoch_loss += losses
        
        scheduler.step()

        # get test loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs, annotations in tqdm(data_loader_test):
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                losses = model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())        
                test_loss += losses

        print(f'Epoch {epoch} loss: {epoch_loss.item()}')
        print(f'Epoch {epoch} test loss: {test_loss.item()}')
        log_file.write(f'{epoch_loss.item()}, {test_loss.item()}\n')

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f'{output_dir}/{epoch}.pth')

    print(f'Training time: {time() - start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=None,
        type=str,
        help="[mobilenetv3 / resnet50]",
    )
    parser.add_argument(
        "--logpath",
        default='logs',
        type=str,
        help="path to store the log file",
    )
    args = parser.parse_args()
    mode = args.mode
    logpath = args.logpath
    main(mode, logpath)