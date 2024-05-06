import random
from typing import List
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader


class PartialImageFolder(ImageFolder):
    def __init__(
        self, 
        root: str,
        transform: torchvision.transforms ,
        input_data_ratio: float,
        label_noise_ratio: float,
        shuffle: bool,
        prev_indices,
        buffer_size
    ):    
        super(PartialImageFolder, self).__init__(root, transform=transform)

        indices = list(range(len(self.samples)))
        dataset_size = int(len(self.samples))

        ################################
        # indices from previous tasks
        if len(prev_indices) > dataset_size:
            raise ValueError("The length of prev_indices cannot be greater than the dataset size.")
        if not all(idx in indices for idx in prev_indices):
            raise ValueError("prev_indices must be a prev of valid dataset indices.")
        
        # select the number of data to be used from previous task
        prev_buffer_indices = prev_indices[:buffer_size]
        
        ################################
        # indices for cur task
        cur_dataset_size = int(len(self.samples) * input_data_ratio)
        cur_indices = [idx for idx in indices if idx not in prev_indices]
        if shuffle:
            random.shuffle(cur_indices)

        # Create the new dataset based on selected_indices  
        self.indices = prev_buffer_indices + cur_indices[:cur_dataset_size]
        self.samples = [self.samples[i] for i in self.indices]
        num_samples = len(self.samples)
        print(num_samples)

        # Introduce label noise
        self.label_noise_indices = random.sample(range(num_samples), int(num_samples * label_noise_ratio))
        self.noisy_labels = {}
        for index in self.label_noise_indices:
            original_label = self.samples[index][1]
            noisy_label = random.choice([l for l in range(len(self.classes)) if l != original_label])
            self.noisy_labels[index] = noisy_label

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        # Use the altered label if the index is in the label noise indices
        if index in self.noisy_labels:
            target = self.noisy_labels[index]

        return sample, target

    def get_indices(self):
        return self.indices
