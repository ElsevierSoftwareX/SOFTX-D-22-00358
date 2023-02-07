


from augmentations_file import *
import torch.nn as nn
import torch.nn.functional as F
from data_parallel import *
from utils import *
from efficientnet_pytorch import EfficientNet
import torchvision



stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transforms = torchvision.transforms.Compose([
    ImgAugTransform(),
    lambda x: PIL.Image.fromarray(x),
    torchvision.transforms.RandomVerticalFlip(),
    tt.RandomHorizontalFlip(),
                         tt.RandomResizedCrop((224,224),interpolation=2),
                         tt.ToTensor(),
                         tt.Normalize(*stats,inplace=True)
                  ])




data_dir = './Data_3/DeepFakes_Dataset_d_7/'


valid_dataset_2 = ImageFolder(r'Data_3/Data_4/valid_data', transform=transforms)

val_ds_2 =valid_dataset_2

from torch.utils.data.dataloader import DataLoader

batch_size=32
PATH = "models_2"


val_dl_2 = DataLoader(val_ds_2, batch_size, num_workers=4, pin_memory=True)

device = get_default_device()
val_dl_2 = DeviceDataLoader(val_dl_2, device)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model,batch) for batch in val_loader]
    return validation_epoch_end(model,outputs)


model_6=torch.load("models_2/74_d56.pth",map_location=device)
model.to(device)
model_6.eval()

print(evaluate(model, val_dl))
print(evaluate_2(model,val_dl_2))
