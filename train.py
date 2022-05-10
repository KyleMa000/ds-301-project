from tqdm import tqdm
from utils.model_utils import get_model_instance_segmentation
import torch
from datasets.mask_dataset import MaskDataset, collate_fn

num_epochs = 20

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(3)
model.to(device)
    
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

dataset = MaskDataset(split='train')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
len_dataloader = len(data_loader)

for epoch in range(num_epochs):
    model.train()
    i = 0    
    epoch_loss = 0
    tqdm_bar = tqdm(data_loader, desc="Epoch {}/{}".format(epoch+1, num_epochs))
    for imgs, annotations in tqdm_bar:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        tqdm_bar.set_description(f'Iteration: {i}/{len_dataloader}, Loss: {losses: .2f}')
        epoch_loss += losses
    
    scheduler.step()
    print(f'Epoch {epoch} loss: {epoch_loss.item()}')

    if epoch % 5 == 0 or epoch == num_epochs - 1:
        torch.save(model.state_dict(), f'logs/faster_rcnn_{epoch}.pth')