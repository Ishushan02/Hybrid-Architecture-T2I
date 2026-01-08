import torch
import torch.nn as nn
# from torchview import draw_graph
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional  as Fn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from io import BytesIO
from IPython.display import Image as IPyImage, display
from PIL import Image
import matplotlib.pyplot as plt
import os
# from piq import ssim
import pandas as pd
from diffusers import AutoencoderDC
import gc
from torchvision import transforms
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    project="HYBRID-T2I",  
    name="vqvae",    
    id="znv3kei1",  
    resume="allow",
)

class VectorQuantizeImage(nn.Module):
    def __init__(self, codeBookDim = 64, embeddingDim = 32, decay = 0.99, eps = 1e-5):
        super().__init__()

        self.codeBookDim = codeBookDim
        self.embeddingDim = embeddingDim
        self.decay = decay
        self.eps = eps
        self.dead_codeBook_threshold = codeBookDim * 0.6

        self.codebook = nn.Embedding(codeBookDim, embeddingDim)
        nn.init.xavier_uniform_(self.codebook.weight.data)

        self.register_buffer('ema_Count', torch.zeros(codeBookDim))
        self.register_buffer('ema_Weight', self.codebook.weight.data.clone())

    def forward(self, x):
        x_reshaped = x.view(-1, self.embeddingDim)

        distance = (torch.sum(x_reshaped**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(x_reshaped, self.codebook.weight.t()))
        
        encoding_indices = torch.argmin(distance, dim=1) 
        encodings = Fn.one_hot(encoding_indices, self.codeBookDim).type(x_reshaped.dtype)
        quantized = torch.matmul(encodings, self.codebook.weight)

        if self.training:
            self.ema_Count = self.decay * self.ema_Count + (1 - self.decay) * torch.sum(encodings, 0)
            
            x_reshaped_sum = torch.matmul(encodings.t(), x_reshaped.detach())
            self.ema_Weight = self.decay * self.ema_Weight + (1 - self.decay) * x_reshaped_sum
            
            n = torch.clamp(self.ema_Count, min=self.eps)
            updated_embeddings = self.ema_Weight / n.unsqueeze(1)
            self.codebook.weight.data.copy_(updated_embeddings)

        
        avg_probs = torch.mean(encodings, dim=0)
        log_encoding_sum = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(log_encoding_sum)

        entropy = log_encoding_sum
        normalized_entropy = entropy / torch.log(torch.tensor(self.codeBookDim, device=x.device))
        diversity_loss = 1.0 - normalized_entropy

        return quantized, encoding_indices, perplexity, diversity_loss
        
        
# vq = VectorQuantizeImage(codeBookDim=64,embeddingDim=32)
# rand = torch.randn(1024,32)
# vq(rand)



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class VecQVAE(nn.Module):
    def __init__(self, inChannels = 1, hiddenDim = 32, codeBookdim = 128, embedDim = 128):
        super().__init__()
        self.inChannels = inChannels
        self.hiddenDim = hiddenDim
        self.codeBookdim = codeBookdim
        self.embedDim = embedDim

        self.encoder = nn.Sequential(
            nn.Conv2d(inChannels, hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            ResidualBlock(hiddenDim),
            ResidualBlock(hiddenDim),
            
            nn.Conv2d(hiddenDim, 2 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            ResidualBlock(2 * hiddenDim),
            ResidualBlock(2 * hiddenDim),
            
            nn.Conv2d(2 * hiddenDim, 4 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(4 * hiddenDim),
            nn.ReLU(inplace=True),
            ResidualBlock(4 * hiddenDim),
            ResidualBlock(4 * hiddenDim),
            
            nn.Conv2d(4 * hiddenDim, embedDim, 1),
        )

        self.vector_quantize = VectorQuantizeImage(codeBookDim=codeBookdim,embeddingDim=embedDim)

        self.decoder = nn.Sequential(
            nn.Conv2d(embedDim, 4 * hiddenDim, 1),
            nn.BatchNorm2d(4 * hiddenDim),
            nn.ReLU(inplace=True),
            
            ResidualBlock(4 * hiddenDim),
            ResidualBlock(4 * hiddenDim),
            nn.ConvTranspose2d(4 * hiddenDim, 2 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            ResidualBlock(2 * hiddenDim),
            ResidualBlock(2 * hiddenDim),
            nn.ConvTranspose2d(2 * hiddenDim, hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            ResidualBlock(hiddenDim),
            ResidualBlock(hiddenDim),
            nn.ConvTranspose2d(hiddenDim, hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, inChannels, 1),
            nn.Sigmoid()
        )

    
    def encoderBlock(self, x, noise_std = 0.15):
        if self.training:
            encodedOut = self.encoder(x)
            encodedOut = encodedOut + torch.randn_like(encodedOut) * noise_std
        else:
            encodedOut = self.encoder(x)

        return encodedOut

    def decoderBlock(self, quantized_vector):
        decodedOut = self.decoder(quantized_vector)
        return decodedOut

    def forward(self, x):
        batch_size, inChannels, height, width = x.shape
        encodedOut = self.encoderBlock(x)
        batch_size, encoded_channel, encoded_height, encoded_width = encodedOut.shape
        
        # print(f"Encoded Shape: {encodedOut.shape}")

        
        vectorize_input = rearrange(encodedOut, 'b c h w -> (b h w) c')
        quantized_vectors, encoding_indices, perplexity, diversity_loss  = self.vector_quantize(vectorize_input)
        codebook_loss = Fn.mse_loss(vectorize_input.detach(), quantized_vectors)
        commitment_loss = Fn.mse_loss(vectorize_input, quantized_vectors.detach())

        quantized_vectors = vectorize_input + (quantized_vectors - vectorize_input).detach()
        # print(f"CodeBook Loss: {codebook_loss} , Commitment Loss: {commitment_loss}")
        # print(f"Quantized SHape: {quantized_vectors.shape}")

        decoder_input = rearrange(quantized_vectors, '(b h w) d -> b d h w', d = encoded_channel, h = encoded_height, w = encoded_width)
        # print(f"Decoded Input SHape: {decoder_input.shape}")
        decodedOut = self.decoderBlock(decoder_input)

        
        return decoder_input, decodedOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss

# VQ = VecQVAE(inChannels = 128, hiddenDim = 256, codeBookdim = 1024, embedDim = 1024)
# test = torch.randn(2, 128, 8, 8)

# quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = VQ(test)
# quantized_latents.shape, decoderOut.shape, codebook_loss, commitment_loss, encoding_indices.shape, perplexity, diversity_loss

DCAEEncoder = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f64c128-in-1.0-diffusers", torch_dtype=torch.float32).to(device).eval()

datasetPath = ""
data = pd.read_csv(datasetPath + "dataset/COCO2017.csv")
# data.head()


class ImageDataset(Dataset):
    def __init__(self, data, rootDir = ""):
        super().__init__()
        self.data = data
        self.rootDir = rootDir
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = os.path.join(self.rootDir, row['imagePath'])
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        # with torch.no_grad():
        #     latents = DCAEEncoder.encode(image).latent
        return image

# imgtxtdata = ImageDataset(data, rootDir=datasetPath)

# img = imgtxtdata.__getitem__(0)
# img.shape

BATCHSIZE = 48
CODEBOOKDIM = 1024
EMBEDDIM = 1025
HIDDENDIM = 256
INPCHANNELS = 128
torchDataset = ImageDataset(data, rootDir=datasetPath)
dataloader = DataLoader(torchDataset, batch_size=BATCHSIZE, shuffle = True)
modelA = VecQVAE(inChannels = INPCHANNELS, hiddenDim = HIDDENDIM, codeBookdim = CODEBOOKDIM, embedDim = EMBEDDIM).to(device)
# lossFn = nn.MSELoss()
optimizerA = torch.optim.Adam([
                    {'params': modelA.encoder.parameters(), 'lr': 2e-4},
                    {'params': modelA.decoder.parameters(), 'lr': 2e-4},
                    {'params': modelA.vector_quantize.parameters(), 'lr': 1e-4}
                ], weight_decay=1e-5)
schedulerA = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizerA, T_0=10, T_mult=2, eta_min=1e-6
            )
epochs = 1000


start_epoch = 0
# baseDir = os.getcwd()#os.path.dirname(__file__)
baseDir = os.path.dirname(__file__)
checkpoint_path = os.path.join(baseDir, "models/vqvae", "vqvae.pt")
print(checkpoint_path)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    modelA.load_state_dict(checkpoint['model_state_dict'])
    optimizerA.load_state_dict(checkpoint['optimizer_state_dict'])
    schedulerA.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    for state in optimizerA.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f"Resuming from epoch {start_epoch}")
else:
    print("Loading pretrained model...")



for each_epoch in range(epochs):
    modelA.train()
    reconstruct_loss = 0.0
    codeb_loss = 0.0
    commit_loss = 0.0
    vqvaeloss = 0.0
    diverse_loss = 0.0
    ssim_loss = 0.0
    
    loop = tqdm(dataloader, f"{each_epoch}/{epochs}")
    perplexities = []

    for images in loop:
        images = images.to(device)
        with torch.no_grad():
            X = DCAEEncoder.encode(images).latent
        
        X = X.to(device)
        Y = X
        # print(X.shape, Y.shape)
    #     break
    # break
        
        quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = modelA(X)

        # ssim_score = ssim(Y, torch.clamp(decoderOut, 0.0, 1.0), data_range=1.0)
        # ssim_score = ssim(Y, decoderOut, data_range=1.0)
        # ssim_loss = 1.0 - ssim_score

        # reconstruction_loss = torch.mean((Y - decoderOut)**2)
        reconstruction_loss = torch.mean(torch.abs(Y - decoderOut))
        
        loss = reconstruction_loss + codebook_loss + 0.2 * commitment_loss + 0.1 * diversity_loss #+ 0.1 * ssim_loss
        vqvaeloss += loss.item()

        
        reconstruct_loss += reconstruction_loss.item()
        diverse_loss += diversity_loss.item()
        codeb_loss += codebook_loss.item()
        commit_loss += commitment_loss.item()
        # ssim_loss += ssim_loss.item()
        perplexities.append(perplexity)
        
        
        optimizerA.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelA.parameters(), max_norm=1.0)
        optimizerA.step()
        loop.set_postfix({"TotalL": f"{vqvaeloss}", "ReconsL": f"{reconstruct_loss}", "CodeL":f"{codeb_loss}",
                          "CommitL":f"{commitment_loss}", "Perplexity":f"{perplexity}", "Diversity Loss":f"{diverse_loss}",# "SSIM Loss":f"{ssim_loss}"
                          })
    #     break
    # break

    average_perplexity = sum(perplexities)/len(perplexities)
    vqvaeloss /= len(dataloader)   
    reconstruct_loss /= len(dataloader)   
    codeb_loss /= len(dataloader)   
    commit_loss /= len(dataloader)   
    diverse_loss /= len(dataloader)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save({
        'epoch': each_epoch,
        'model_state_dict': modelA.module.state_dict(),
        'optimizer_state_dict': optimizerA.state_dict(),
        'scheduler_state_dict': schedulerA.state_dict()
    }, checkpoint_path)
    wandb.log({
        "Epoch": each_epoch,
        "VQVAE LR": optimizerA.param_groups[0]['lr'],
        "VQVAE Loss": vqvaeloss,
        "Reconstruction Loss": reconstruct_loss,
        "Codebook Loss": codeb_loss,
        "Commitment Loss": commit_loss,
        "Diversity Loss": diverse_loss,
        "Perplexity": average_perplexity,
       # "SSIM Loss":ssim_loss,
    })
    schedulerA.step()
 