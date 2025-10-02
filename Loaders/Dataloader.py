import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
import random


class DataLoaderLite:
    def __init__(self, B, process_rank = 0, num_processes = 1, shuffle = True):
        self.B = B

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.imgfolder = "frames" 
        self.files = sorted([f for f in os.listdir(self.imgfolder) if f.endswith(".jpg")])

        print(f"loaded {len(self.files)} images from {self.imgfolder}")
        self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                  std=[0.229, 0.224, 0.225])])
        
        self.target = np.loadtxt("target.csv", delimiter=",", dtype=int)
        self.target = torch.from_numpy(self.target).long() # pytorch wants long tensors for labels
        assert len(self.files) == len(self.target), "number of images and targets must match"
        
        self.indices = list(range(len(self.files)))
        if shuffle:
            random.shuffle(self.indices)

        self.current_position = self.process_rank * B # process_rank starts at 0

    def next_batch(self):
        B = self.B

        indices = self.indices[self.current_position : self.current_position + B]
        clip_files = [self.files[i] for i in indices]

        frames = [self.transform(Image.open(os.path.join(self.imgfolder, f)).convert("RGB")) for f in clip_files]
        x = torch.stack(frames, dim=0) # (B, 3, 224, 224)
        y = self.target[indices] # (B,)
        
        # advance the current position
        self.current_position += B * self.num_processes

        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * self.num_processes) > len(self.files):
            self.current_position = self.process_rank * B
            random.shuffle(self.indices)
        return x, y