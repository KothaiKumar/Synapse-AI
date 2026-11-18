import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_genomics(path):
    df=pd.read_csv(path,sep="\t",comment="!")
    df=df.set_index(df.columns[0])
    df=df.T
    scaler=StandardScaler()
    X=scaler.fit_transform(df.values)
    pca=PCA(n_components=128)
    X=pca.fit_transform(X)
    return torch.tensor(X,dtype=torch.float32)

def create_class_genomics(genomic_tensor):
    c0=genomic_tensor.mean(0)
    c1=genomic_tensor[:len(genomic_tensor)//2].mean(0)
    c2=genomic_tensor[len(genomic_tensor)//2:].mean(0)
    c3=torch.zeros_like(c0)
    return torch.stack([c0,c1,c2,c3])

class MRIDataset(Dataset):
    def __init__(self, root, transform, split="Training"):
        self.samples = []
        self.transform = transform

        split_path = os.path.join(root, split)
        self.classes = sorted(os.listdir(split_path))
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        for c in self.classes:
            class_dir = os.path.join(split_path, c)

            for f in os.listdir(class_dir):
                full_path = os.path.join(class_dir, f)

                if os.path.isfile(full_path) and f.lower().endswith((".jpg",".jpeg",".png")):
                    self.samples.append((full_path, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


class GenomicEncoder(nn.Module):
    def __init__(self,input_dim,embed_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512,embed_dim),
            nn.ReLU()
        )
    def forward(self,x):
        return self.net(x)

class RiskHead(nn.Module):
    def __init__(self,embed_dim=128):
        super().__init__()
        self.fc=nn.Linear(embed_dim,1)
    def forward(self,g):
        risk=torch.sigmoid(self.fc(g))
        conditioned = g * (0.5 + 0.5*risk)
        return conditioned,risk

class MRIEncoder(nn.Module):
    def __init__(self,embed_dim=128):
        super().__init__()
        backbone=mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features=backbone.features
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.proj=nn.Linear(1280,embed_dim)
    def forward(self,x):
        x=self.features(x)
        x=self.pool(x).view(x.size(0),-1)
        x=self.proj(x)
        return x

class RiskGatedFusion(nn.Module):
    def __init__(self,embed_dim=128,drop_prob=0.3):
        super().__init__()
        self.norm_v=nn.LayerNorm(embed_dim)
        self.norm_g=nn.LayerNorm(embed_dim)
        self.gate_net=nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim,embed_dim),
            nn.Sigmoid()
        )
        self.out_norm=nn.LayerNorm(embed_dim)
        self.drop_prob=drop_prob

    def forward(self,v,g):
        v=self.norm_v(v)
        g=self.norm_g(g)

        if self.training:
            mask=(torch.rand(v.size(0),1,device=v.device)>self.drop_prob).float()
            g=g*mask

        gate=self.gate_net(g)
        fused=gate*v+(1-gate)*g
        fused=self.out_norm(fused+v)
        return fused

    
class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model
        self.gradients=None
        self.activations=None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self,module,input,output):
        self.activations=output

    def backward_hook(self,module,grad_in,grad_out):
        self.gradients=grad_out[0]

    def generate(self,input_image,genome,label):
        self.model.zero_grad()
        output,_,_,_=self.model(input_image,genome)
        loss=output[:,label].sum()
        loss.backward()

        grads=self.gradients.mean(dim=[2,3],keepdim=True)
        cam=(grads*self.activations).sum(dim=1)
        cam=torch.relu(cam)
        cam=cam.squeeze().cpu().detach().numpy()
        cam=(cam-cam.min())/(cam.max()+1e-8)
        return cam


class main(nn.Module):
    def __init__(self,genomic_input_dim,num_classes=4):
        super().__init__()
        self.mri=MRIEncoder()
        self.genomic=GenomicEncoder(genomic_input_dim)
        self.risk=RiskHead()
        self.fusion=RiskGatedFusion()
        self.classifier=nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )
    def forward(self,image,genome):
        v=self.mri(image)
        g=self.genomic(genome)
        g,r=self.risk(g)
        fused=self.fusion(v,g)
        out=self.classifier(fused)
        return out,r,v,fused
class BioVisualGradCAM:
    def __init__(self,model,target_layer):
        self.model=model
        self.gradients=None
        self.activations=None
        target_layer.register_forward_hook(self.f_hook)
        target_layer.register_backward_hook(self.b_hook)
    def f_hook(self,module,input,output):
        self.activations=output
    def b_hook(self,module,grad_in,grad_out):
        self.gradients=grad_out[0]
    def generate(self,img,genome,label):
        self.model.zero_grad()
        out,risk,_,_=self.model(img,genome)
        score=out[:,label]*risk.squeeze()
        score.sum().backward()
        grads=self.gradients.mean(dim=[2,3],keepdim=True)
        cam=(grads*self.activations).sum(dim=1)
        cam=torch.relu(cam)
        cam=cam.squeeze().detach().cpu().numpy()
        cam=(cam-cam.min())/(cam.max()+1e-8)
        return cam
balance_criterion = nn.MSELoss()
if __name__=="__main__":

    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    genomic_raw=load_genomics("GSE4290_series_matrix.txt")
    class_genomics=create_class_genomics(genomic_raw).to(device)

    train_dataset=MRIDataset("mri_dataset",transform,split="Training")
    test_dataset=MRIDataset("mri_dataset",transform,split="Testing")

    train_size=int(0.85*len(train_dataset))
    val_size=len(train_dataset)-train_size

    train_set,val_set=torch.utils.data.random_split(
        train_dataset,[train_size,val_size]
    )
    test_set=test_dataset

    train_loader=DataLoader(train_set,batch_size=16,shuffle=True)
    val_loader=DataLoader(val_set,batch_size=16,shuffle=False)
    test_loader=DataLoader(test_set,batch_size=16,shuffle=False)


    model=main(genomic_raw.shape[1]).to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-5)

    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]


    for epoch in range(30):

        model.train()
        running_loss=0
        correct=0
        total=0

        for imgs,labels in train_loader:
            imgs=imgs.to(device)
            labels=labels.to(device)
            genome_batch=class_genomics[labels]

            optimizer.zero_grad()
            outputs,_,v,fused = model(imgs,genome_batch)
            cls_loss = criterion(outputs,labels)
            balance_loss = balance_criterion(v,fused)
            loss = cls_loss + 0.1 * balance_loss
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            preds=outputs.argmax(1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)

        train_loss.append(running_loss/len(train_loader))
        train_acc.append(correct/total)

        model.eval()
        v_loss=0
        v_correct=0
        v_total=0

        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs=imgs.to(device)
                labels=labels.to(device)
                genome_batch=class_genomics[labels]
                outputs,_,_,_=model(imgs,genome_batch)
                loss=criterion(outputs,labels)

                v_loss+=loss.item()
                preds=outputs.argmax(1)
                v_correct+=(preds==labels).sum().item()
                v_total+=labels.size(0)

        val_loss.append(v_loss/len(val_loader))
        val_acc.append(v_correct/v_total)

        print(epoch,train_loss[-1],train_acc[-1],val_loss[-1],val_acc[-1])




    plt.plot(train_loss,label="train_loss")
    plt.plot(val_loss,label="val_loss")
    plt.legend()
    plt.show()

    plt.plot(train_acc,label="train_acc")
    plt.plot(val_acc,label="val_acc")
    plt.legend()
    plt.show()

    target_layer = model.mri.features[-2]
    gradcam=GradCAM(model,target_layer)

    img,label = test_dataset[0]
    img=img.unsqueeze(0).to(device)
    genome_batch=class_genomics[label].unsqueeze(0)

    heatmap=gradcam.generate(img,genome_batch,label)

    plt.imshow(heatmap,cmap="jet")
    plt.title("Multimodal GradCAM")
    plt.show()

    target_layer=model.mri.features[-1]
    biocam=BioVisualGradCAM(model,target_layer)

    img,label = test_dataset[0]
    img=img.unsqueeze(0).to(device)
    genome_batch=class_genomics[label].unsqueeze(0)

    heatmap=biocam.generate(img,genome_batch,label)

    plt.imshow(heatmap,cmap="jet")
    plt.title("BioVisual Risk GradCAM")
    plt.show()



    model.eval()
    y_true=[]
    y_pred=[]

    with torch.no_grad():
        for imgs,labels in test_loader:
            imgs=imgs.to(device)
            labels=labels.to(device)
            genome_batch=class_genomics[labels]
            out,_,_,_=model(imgs,genome_batch)
            preds=out.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm=confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()

    print(classification_report(y_true,y_pred))
    torch.save(model.state_dict(), "brain_tumor_model.pth")