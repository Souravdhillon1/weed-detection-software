import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class WeedDataset:
    def __init__(self, dataset_path, batch_size=32):
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor()
        ])

        # Load dataset
        self.dataset = datasets.ImageFolder(root=dataset_path, transform=self.transform)

        # Create DataLoader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Print class names to verify
        print("Classes:", self.dataset.classes)
    def get_dataset(self):
        return self.dataset


    def show_sample(self, class_name):
        if class_name not in self.dataset.classes:
            print(f"Class '{class_name}' not found!")
            return

        class_index = self.dataset.classes.index(class_name)
        found = False

        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            if label == class_index:
                plt.imshow(img.permute(1, 2, 0))
                plt.title(self.dataset.classes[label])
                plt.show()
                found = True
                break

        if not found:
            print(f"No images found for class: {class_name}")


# Test again
#dataset_path = r"E:\\2\\Weeds dataset Uploaded\\Weeds dataset Uploaded"
#weed_dataset=WeedDataset(dataset_path)
#weed_dataset.show_sample('Nutgrass')

