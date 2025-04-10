import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from weed_dataset import WeedDataset
from weed_dataset_splitter import WeedDatasetSplitter

class WeedModel:
    def __init__(self,num_classes,device='cpu'):
        self.model=models.resnet18(pretrained=True)
        num_ftrs=self.model.fc.in_features
        self.model.fc=nn.Linear(num_ftrs,num_classes)
        self.device=device
        self.model.to(self.device)
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer=optim.Adam(self.model.parameters(),lr=0.001)

    def train(self,train_dataloader,num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss=0.0
            for inputs,labels in train_dataloader:
                inputs,labels=inputs.to(self.device),labels.to(self.device)
                self.optimizer.zero_grad()
                outputs=self.model(inputs)
                loss=self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader)}")
    def evaluate(self,dataloader):
        self.model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for inputs,labels in dataloader:
                inputs,labels=inputs.to(self.device),labels.to(self.device)
                outputs=self.model(inputs)
                _,predicted=torch.max(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
dataset_path = r"E:\\2\\Weeds dataset Uploaded\\Weeds dataset Uploaded"
weed_Dataset=WeedDataset(dataset_path).get_dataset()
data_splitter=WeedDatasetSplitter(weed_Dataset,train_ratio=0.8,val_ratio=0.1,test_ratio=0.1)
train_dataloader,val_dataloader,test_dataloader=data_splitter.get_dataloaders(batch_size=32)
num_classes=len(weed_Dataset.classes)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cuda availability:{torch.cuda.is_available()}")

weed_model=WeedModel(num_classes,device)
weed_model.train(train_dataloader,num_epochs=10)
validation_accuracy=weed_model.evaluate(val_dataloader)
print(f"Validation Accuracy: {validation_accuracy:.2f}%")
test_accuracy = weed_model.evaluate(test_dataloader)
print(f"Test Accuracy: {test_accuracy:.2f}%")
#
