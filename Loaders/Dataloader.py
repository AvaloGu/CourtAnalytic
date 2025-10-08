import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
import random
from itertools import groupby


class DataLoaderLite:
    def __init__(self, process_rank = 0, num_processes = 1, train = True):

        self.B = 32
        self.batch_sizes = [32, 48, 64, 96, 128]
        self.train = train

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.imgfolder = "frames" 
        self.files = sorted([f for f in os.listdir(self.imgfolder) if f.endswith(".jpg")])

        print(f"loaded {len(self.files)} images from {self.imgfolder}")
        self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                  std=[0.229, 0.224, 0.225])])
        
        self.target = np.loadtxt("target_multi_class.csv", delimiter=",", dtype=int)
        self.target = torch.from_numpy(self.target).long() # pytorch wants long tensors for labels
        assert len(self.files) == len(self.target), "number of images and targets must match"
        
        self.indices_original = list(range(len(self.files)))
        self.indices = self.indices_original.copy()

        # self.current_position = self.process_rank # process_rank starts at 0


    def next_batch(self):
        # this is inefficient dataloader due to deleting elements from a dynamic array (python list)
        # but we do want the randomness in both batch size and the segment we are fetching from the video
        # this helps in both convergence and transfering to stage 2.
        # DDP won't work

        random_index = random.randrange(len(self.indices))

        random_batch_size = random.choice(self.batch_sizes)
        self.B = random_batch_size

        # python handle out of bound slicing gracefully
        indices = self.indices[random_index: random_index + random_batch_size]
        del self.indices[random_index: random_index + random_batch_size]

        clip_files = [self.files[i] for i in indices]

        frames = [self.transform(Image.open(os.path.join(self.imgfolder, f)).convert("RGB")) for f in clip_files]
        x = torch.stack(frames, dim=0) # (B, 3, 224, 224)
        y = self.target[indices] # (B,)

        if len(self.indices) == 0:
            self.indices = self.indices_original.copy()
        
        # advance the current position
        # self.current_position += self.num_processes

        # if loading the next batch would be out of bounds, reset for next epoch
        # if self.current_position >= len(self.indices):
        #     self.current_position = self.process_rank
        #     self.re_randomize()
            
        return x, y
    

class DataLoaderStage2:
    def __init__(self, process_rank = 0, num_processes = 1):

        self.num_examples = 0

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.imgfolder = "frames" 
        self.files = sorted([f for f in os.listdir(self.imgfolder) if f.endswith(".jpg")])

        print(f"loaded {len(self.files)} images from {self.imgfolder}")
        self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                  std=[0.229, 0.224, 0.225])])
        
        target = np.loadtxt("target_stage2.csv", dtype=str)
        # the point separator 'z' is encoded as 13, vocab_size is 13
        stoi = {ch:i for i, ch in enumerate(np.sort(np.unique(target)))}
        encode = [stoi[s] for s in target]
        self.target = [] # a list of lists
        for k, g in groupby(encode, lambda x: x==13):
            if not k:
                self.target.append(list(g))
        
        frame_label = np.loadtxt("target_multi_class.csv", delimiter=",", dtype=int)
        frame_label = torch.from_numpy(frame_label).long()
        assert len(self.files) == len(frame_label), "number of images and targets must match"
    
        # we only care about whether a frame is during point or out of point here
        for i in range(4):
            frame_label[frame_label == i] = 5
        label_original = frame_label[:-1]
        label_shift_by_1 = frame_label[1:]
        diff = label_shift_by_1 - label_original
        in_point_out_point_indices = (torch.where((diff==1) | (diff==-1)))[0]

        # indices of the in point frames
        self.indices = [] # a list of lists, each sublist is a point
        for a, b in zip(in_point_out_point_indices[:-1:2], in_point_out_point_indices[1::2]):
            self.indices.append(list(range(a.item(), b.item())))

        # TODO: you might want to add a preprocessing step to check for this condition
        assert len(self.indices) == len(self.target), "number of points must match number of points in target"

        self.current_position = self.process_rank # process_rank starts at 0

    def next_batch(self):
        B = 16 # for experimenting

        indices = self.indices[self.current_position]
        indices = indices[:B]
        clip_files = [self.files[i] for i in indices]

        frames = [self.transform(Image.open(os.path.join(self.imgfolder, f)).convert("RGB")) for f in clip_files]
        x = torch.stack(frames, dim=0) # (B, 3, 224, 224)
        y = self.target[self.current_position]
        self.num_examples = len(y)
        y = torch.tensor(y) # (Num of shots in this point,)
        
        # advance the current position
        self.current_position += self.num_processes

        # if loading the next batch would be out of bounds, reset
        if self.current_position >= len(self.target):
            self.current_position = self.process_rank
        return x, y # (B, 3, 224, 224), (Num of shots in this point,)