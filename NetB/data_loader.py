import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch
import random
from scipy import io

class DatasFolder(Dataset):
    def __init__(self, root, phase):
        super(DatasFolder, self).__init__()
        self.max_val = 2**16 # 16bit digitizer
        self.phase = phase
        self.phase_folder = phase
        self.root = root
        self.file_paths = os.listdir(os.path.join(self.root, "data", self.phase_folder))
        self.file_paths.sort()

        self.target_paths = os.listdir(os.path.join(self.root, "ground_truth", self.phase_folder))
        self.target_paths.sort()

        assert len(self.file_paths) == len(self.target_paths), "The number of images and masks are different."

        self.data_paths = []
        for i in range(len(self.file_paths)):
            data_path = (os.path.join(self.root, "data", self.phase_folder, self.file_paths[i]),
                         os.path.join(self.root, "ground_truth", self.phase_folder, self.target_paths[i]))
            self.data_paths.append(data_path)

    def __getitem__(self, index):
        file_path, target_path = self.data_paths[index]  # Random index
        data = io.loadmat(file_path)
        target = io.loadmat(target_path)

        data = np.array(data['data']).astype(np.float32)
        target = np.array(target['ground_truth']).astype(np.float32)

        center_v = 96.0
        max_v = 20.0

        if self.phase == 'train' or self.phase =='val':

            # Random brightness adjustment
            if random.random() < 0.5:
                center_v = center_v + random.randrange(-3, 3)

            if random.random() < 0.5:
                max_v = max_v + random.randrange(-3, 3)

            data = (data - center_v) / max_v
            target = (target - center_v) / max_v

            data = F.to_pil_image(data, mode='F')
            target = F.to_pil_image(target, mode='F')
            # Random cropping
            i, j, h, w = T.RandomCrop.get_params(data, output_size=[128, 128])

            data = F.crop(data, i, j, h, w)
            target = F.crop(target, i, j, h, w)

            # Random horizontal flipping
            if random.random() < 0.5:
                data = F.hflip(data)
                target = F.hflip(target)

            # Random vertical flipping
            if random.random() < 0.5:
                data = F.vflip(data)
                target = F.vflip(target)

            data, target = np.array(data), np.array(target)
        else:
            data = (data-center_v)/max_v
            target = (target-center_v)/max_v


        transform = T.Compose([T.ToTensor()])
        data, target = transform(data), transform(target)

        return data, target

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)


def get_loader(dataset_path, phase="train", shuffle=True, batch_size=1, num_workers=2):
    """Builds and returns Dataloader."""

    dataset = DatasFolder(root=dataset_path, phase=phase)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
