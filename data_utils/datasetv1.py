import torch
import os
import numpy as np
from diffsynth import VideoData
from einops import repeat
import torchvision.transforms as transforms


class ProcgenDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, game, height, width, num_frames):
        self.data_folder=os.path.join(base_path, game)
        self.items=np.load(os.path.join(self.data_folder,'info_new.npz'))['items']
        self.height=height
        self.width=width
        self.num_frames=num_frames

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        episode_idx, frame_idx=self.items[idx]
        episode=np.load(os.path.join(self.data_folder, f'episode_{episode_idx}.npz'))
        video=episode['obs'][frame_idx:frame_idx+self.num_frames]                   # [9, 64, 64, 3] (0,255) np.float32
        action=episode['action'][frame_idx:frame_idx+self.num_frames-1]             # [8, ]  np.int64

        # transform video
        video=torch.from_numpy(video).to(dtype=torch.bfloat16).permute(0,3,1,2)
        video=torch.nn.functional.interpolate(video,size=(self.height,self.width),mode='bilinear',align_corners=False)
        video=video*2.0/255.0 - 1.0
        video=video.permute(1,0,2,3)                                                # [3, 9, 64, 64] (-1,1) torch.bf16

        # transform action
        action=action.astype(np.int32)
        return {'video':video, 'prompt':action}
    
if __name__ == "__main__":
    dataset=ProcgenDataset(base_path='./../data/procgen',
                           game='ninja',
                           height=256,
                           width=256,
                           num_frames=9)
    import ipdb; ipdb.set_trace()