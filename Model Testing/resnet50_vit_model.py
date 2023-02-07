import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import copy
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
#from torchsummary import summary
import torchvision.models as models
from torch.autograd import Variable
from vision_transformer_pytorch import VisionTransformer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#positions are here learned not fixedly added
#instead of linear leaye, conv2d can be added
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 512, patch_size: int = 4, emb_size: int = 768, img_size: int = 28):#in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224
        self.patch_size = patch_size
        super().__init__()
        """
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        """
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size * patch_size * in_channels, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1,emb_size))
                
    def forward(self, x: Tensor):
        b,c,h,w = x.shape
        #x = self.projection(x)
        x = rearrange(x,"b c (h s1) (w s2) -> b (h w) (s1 s2 c)", s1=self.patch_size, s2=self.patch_size)
        x = self.linear(x)
        cls_tokens = repeat(self.cls_token,'() n e -> b n e', b=b)#repeat the cls tokens for all patch set in 
        x = torch.cat([cls_tokens,x],dim=1)
        x+=self.positions
        return x
        
class multiHeadAttention(nn.Module):
  def __init__(self, emb_size: int=768, heads: int=8, dropout: float=0.0):
    super().__init__()
    self.heads = heads
    self.emb_size = emb_size
    self.query = nn.Linear(emb_size,emb_size)
    self.key = nn.Linear(emb_size,emb_size)
    self.value = nn.Linear(emb_size,emb_size)
    self.drop_out = nn.Dropout(dropout)
    self.projection = nn.Linear(emb_size,emb_size)

  def forward(self,x):
    #splitting the single input int number of heads
    queries = rearrange(self.query(x),"b n (h d) -> b h n d", h = self.heads)
    keys = rearrange(self.key(x),"b n (h d) -> b h n d", h = self.heads)
    values = rearrange(self.value(x),"b n (h d) -> b h n d", h = self.heads)

    attention_maps = torch.einsum("bhqd, bhkd -> bhqk",queries,keys)
    scaling_value = self.emb_size**(1/2)
    attention_maps = F.softmax(attention_maps,dim=-1)/scaling_value
    attention_maps = self.drop_out(attention_maps)##might be deleted

    output = torch.einsum("bhal, bhlv -> bhav",attention_maps,values)
    output  = rearrange(output,"b h n d -> b n (h d)")
    output = self.projection(output)
    return output

class residual(nn.Module):
  def __init__(self,fn):
    super().__init__()
    self.fn = fn
  def forward(self,x):
    identity = x
    res = self.fn(x)
    out = res + identity
    return out

class mlp(nn.Sequential):#multi layer perceptron
  def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerBlock(nn.Sequential):
  def __init__(self,emb_size:int = 768,drop_out:float=0.0):
    super().__init__(
        residual(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                multiHeadAttention(emb_size),
                nn.Dropout(drop_out)
            )
        ),
        residual(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                mlp(emb_size),
                nn.Dropout(drop_out)
            )
        )
    )

class Transformer(nn.Sequential):
  def __init__(self,loops:int =12):
    super().__init__(
        *[TransformerBlock() for _ in range(loops)]
    )

class Classification(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)
    def forward(self, x: Tensor):
        x = reduce(x,'b n e -> b e', reduction='mean')
        x = self.norm(x)
        output = self.linear(x)
        print(output)
        return output
        
        
class VIT(nn.Module):
  def __init__(self,resnetM,emb_size: int=768,drop_out: float=0.0, n_classes:int = 2,in_channels:int=512,patch_size:int=4,image_size:int=28):
    super().__init__()
    self.resnetM = resnetM
    self.PatchEmbedding = PatchEmbedding(in_channels,patch_size,emb_size,image_size)
    self.Transformer = Transformer()
    self.Classification = Classification(n_classes=2)
  def forward(self,x):
    resnetOutput = self.resnetM(x)
    #print(resnetOutput.shape)
    patchEmbeddings = self.PatchEmbedding(resnetOutput)
    transformerOutput = self.Transformer(patchEmbeddings)
    classificationOutput = self.Classification(transformerOutput)
    #output = F.log_softmax(classificationOutput, dim=1)
    output = F.softmax(classificationOutput, dim=1)    
    #output = torch.softmax(classificationOutput.squeeze(),0)
    return output
    #return classificationOutput
################################

