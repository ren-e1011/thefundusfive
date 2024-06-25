from torch.utils.data import Dataset
from torchvision import transforms
from torch import tensor 
from timm.data import create_transform
# ret model was originally trained on imagenet so bw images should be scaled accordingly
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
from PIL import Image

class DataSet(Dataset):

    def __init__(self,data_path, data_params, train=False,device='cpu'):

        self.training = train

        self.path = data_path
        # self.sample_list = os.listdir(self.path)
        # Thumbs.db
        self.sample_list = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] == '.png'] 

        self.data_params = data_params

        self.transform_list = [
                            transforms.Resize(size=self.data_params.out_size,interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean= self.data_params.mean_imgnet, std=self.data_params.std_imgnet) 
                            ]

        self.device = device 

        self.target_dict = {'A':0,'D':1,'G':2,'N':3}

    def __len__(self):
        return len(self.sample_list)
    
    # needs to resize and normalize at every sample
    # train should also augment with flip, crop
    def transform(self, X,index):

        if self.training:
            # todo add transform_list to crop, etc augmentation
            data_transform = transforms.Compose(self.transform_list)
        
        else:
            data_transform = transforms.Compose(self.transform_list)

        return data_transform(X)
    
    def __getitem__(self, index):
        
        im_path = self.sample_list[index]
        target = im_path.split('_')[1][0] # 599_N.png
        y = tensor(self.target_dict[target]) # N -> 3
        # RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
        # https://stackoverflow.com/questions/58496858/pytorch-runtimeerror-the-size-of-tensor-a-4-must-match-the-size-of-tensor-b
        im = Image.open(os.path.join(self.path,im_path)).convert('RGB') 
        X = self.transform(im,index) # pass index for debugging 

        X,y = X.to(self.device), y.to(self.device)

        return X, y 
        