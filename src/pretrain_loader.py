import torch
from torch.utils.data import Dataset, DataLoader
from pretrain_params import training_label, clip_dir, num_frames, image_size, test_label
import torchvision.transforms as transforms
from torchvision.io import decode_image
import os



    


class DFEW_pretrain(Dataset):
    def __init__(self,mode):
        self.mode = mode
        assert self.mode in ['train', 'validation']
        
        if self.mode == 'train':
            self.training_label = training_label
        if self.mode == 'validation':
            self.training_label = test_label
        
        self.clip_dir = clip_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.clips = self.prepare_file()
        self.data = self.prepare_data()
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),  # float \in [0, 1]
            transforms.Resize((self.image_size, self.image_size)),             
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize
            ])

    def sample_frames(self, frames: list):
        assert type(frames) == list
        total_frames = len(frames)
        if total_frames == self.num_frames:
            return frames
        elif total_frames < self.num_frames:
            # repeat the last frame, add the last frame until the number of frames is 16
            return frames + [frames[-1]] * (self.num_frames - total_frames)
        else:
            return frames[::total_frames//self.num_frames][:self.num_frames]    

    def prepare_file(self):
        # list of ([clip_path, clip_path, ], label)
        clips = []

        with open(self.training_label, 'r') as f:
            # igore the header
            f.readline()
            # from the second line
            lines = f.readlines()
            for line in lines:
                clip_paths = []
                clip_name, label = line.strip().split(',')
                while len(clip_name) < 5:
                    clip_name = '0' + clip_name
                # add the full path
                clip_path = os.path.join(self.clip_dir, clip_name)
                # read the frames in the clip_path
                frames = os.listdir(clip_path)

                # if the clip has less than 16 frames, skip it
                if len(frames) < self.num_frames:
                    continue
                # sort the frames
                frames.sort()
                
                # select 16 frames from frames
                sampled_frames = self.sample_frames(frames)

                # add the full path of each frame
                for frame in sampled_frames:
                    clip_paths.append(os.path.join(clip_path, frame))
                # add the clip_path and label to the clips list
                clips.append((clip_paths, int(label)))
        return clips



    def prepare_data(self):
        data = []
        for clip_paths, label in self.clips:
            clip = []
            for frame_path in clip_paths:
                frame = decode_image(frame_path, mode='RGB')
                clip.append(frame)
            clip = torch.stack(clip, dim=0)
            data.append((clip, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip, label = self.data[idx]
        clip = self.transform(clip)
        return clip, label
    
if __name__ == '__main__':
    # test the DFEW_pretrain
    dataset = DFEW_pretrain("train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
    for i, (clip, label) in enumerate(dataloader):
        # torch.Size([2, 16, 3, 224, 224])
        # 2 clips, 16 frames, 3 channels, 224x224
        print(clip.shape)
        # tensor([5, 3])
        print(label)
        break