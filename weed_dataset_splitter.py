import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from weed_dataset import WeedDataset
class WeedDatasetSplitter:
    def __init__(self,dataset,train_ratio=0.7,val_ratio=0.15,test_ratio=0.15):
        self.dataset=dataset
        self.train_size=int(train_ratio*len(dataset))
        self.val_size=int(val_ratio*len(dataset))
        self.test_size=len(dataset)-self.train_size-self.val_size
        self.train_dataset,self.val_dataset,self.test_dataset=data.random_split(dataset,[self.train_size,self.val_size,self.test_size])

    def get_dataloaders(self,batch_size=32):
        train_dataloader=data.DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True)
        val_dataloader=data.DataLoader(self.val_dataset,batch_size=batch_size,shuffle=False)
        test_dataloader=data.DataLoader(self.test_dataset,batch_size=batch_size,shuffle=False)
        return train_dataloader,val_dataloader,test_dataloader

#dataset_path = r"E:\\2\\Weeds dataset Uploaded\\Weeds dataset Uploaded"
#weed_Dataset=WeedDataset(dataset_path).get_dataset()
#dataset_splitter=WeedDatasetSplitter(weed_Dataset)
#train_loader,val_loader,test_loader=dataset_splitter.get_dataloaders(batch_size=32)
#print(f"Train size: {len(dataset_splitter.train_dataset)}")
#print(f"Validation size: {len(dataset_splitter.val_dataset)}")
#print(f"Test size: {len(dataset_splitter.test_dataset)}")
print(torch.cuda._get_device(0))
