import torch
from utils.eval_utils import evaluate
from datasets.mask_dataset import MaskDataset, collate_fn
from utils.model_utils import get_model_instance_segmentation

checkpoint_path = 'logs/faster_rcnn_19.pth'

dataset = MaskDataset(split='test')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model_instance_segmentation(3)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
model.to(device)

evaluate(model, data_loader, device=device)