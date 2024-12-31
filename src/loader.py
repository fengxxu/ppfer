import os
import random 
import time
ALLSTART = time.time()

from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch
from torchvision import transforms
from pytorch_wavelets import DWTForward, DWTInverse

from train_params import train_identity_file, train_emotion_file, s1_video_clip_dir, s2_video_clip_dir, s3_video_clip_dir, test_identity_file, test_emotion_file, num_frames, image_size

# class WaveletTransform:
#     def __init__(self, wave='db1', level=1):
#         self.dwt = DWTForward(J=level, wave=wave)
#         self.idwt = DWTInverse(wave=wave)

#     def __call__(self, x):
#         if x.ndimension() == 3:
#             x = x.unsqueeze(0)
#         low, high = self.dwt(x)
#         return low, high


class Cremad(Dataset):
    def __init__(self, mode, stage):
        assert mode in ['train', 'validation', 'infer_train']
        assert stage in [1,2,3]
        self.mode=mode
        if stage == 1:
            self.clip_dir = s1_video_clip_dir
        elif stage == 2:
            self.clip_dir = s2_video_clip_dir
        else:
            self.clip_dir = s3_video_clip_dir
        self.num_frames = num_frames
        self.image_size = image_size
        if self.mode == 'train':
            self.identity_file = train_identity_file
            self.emotion_file = train_emotion_file
            self.transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.float),  # float \in [0, 1]
                transforms.Resize((self.image_size, self.image_size)),                 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            ])

        if self.mode == 'infer_train':
            self.identity_file = train_identity_file
            self.emotion_file = train_emotion_file
            self.transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.float),  # float \in [0, 1]
                transforms.Resize((self.image_size, self.image_size)),                 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize
            ])
            
        if self.mode == 'validation':
            self.identity_file = test_identity_file
            self.emotion_file = test_emotion_file
            self.transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.float),  # float \in [0, 1]
                transforms.Resize((self.image_size, self.image_size)),                 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize
            ])


        self.data = self.prepare_data()


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

    def read_csv(self, file_path):
        data = {}
        with open(file_path, 'r') as f:
            # igore the header
            f.readline()
            # from the second line
            lines = f.readlines()
            for line in lines:
                clip_name, label = line.strip().split(',')
                while len(clip_name) < 5:
                    clip_name = '0' + clip_name
                data[clip_name] = label
        return data

    def prepare_data(self):
        """
        data: list of ([clip_path, clip_path, ... x16], [identity_label, emotion_label])
        
        """
        data=[]
        dir_identity = self.read_csv(self.identity_file)
        dir_emotion = self.read_csv(self.emotion_file)
        for clip_name in dir_identity.keys():
            clip_path = os.path.join(self.clip_dir, clip_name)
            frames_path = os.listdir(clip_path)
            frames = self.sample_frames(frames_path)
            frames = [os.path.join(clip_path, frame) for frame in frames]
            identity_label = dir_identity[clip_name]
            emotion_label = dir_emotion[clip_name]
            data.append((frames, [identity_label, emotion_label]))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frames, labels = self.data[idx]
        images = []
        for frame in frames:
            image = decode_image(frame, mode='RGB')
            images.append(image)
        images = torch.stack(images)
        images = self.transform(images)
        return images, torch.tensor([int(i) for i in labels])
    


    

if __name__ == '__main__':
    
    # start = time.time()
    # print("overhead: ",start-ALLSTART) # 2.8809587955474854
    # dataset = Cremad('train')
    # print(len(dataset))
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    # print("Train set: ", time.time()-start)
    # dataset = Cremad('validation')
    # print(len(dataset))
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    # print("Test set: ", time.time()-start) # 6.841085910797119

    # test wavelet transform
    dataset = Cremad('train')
    dwt = DWTForward(J=1, wave='haar', mode='zero')
    dl = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for imgs, labels in dl:
        print(imgs.shape)
        print(labels)
        imgs = imgs.view(-1, 3, 224, 224)
        low, high = dwt(imgs)
        print(low.shape)
        print(high.__len__())
        for i in high:
            print(i.shape)

        low_reshape = low.view(2, 16, 3, *low.shape[2:])    
        high_reshape = [h.view(2, 16, 3, *h.shape[2:]) for h in high]
        high_stack = torch.stack([high_reshape[0][:,:,0,:,:],
                                  high_reshape[0][:,:,1,:,:],
                                  high_reshape[0][:,:,2,:,:]], dim=1)
        print(low_reshape.shape)
        print(high_reshape[0].shape)
        print(high_stack.shape)
        break


        


