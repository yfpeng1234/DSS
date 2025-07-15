import torch
import os
import numpy as np
from diffsynth import VideoData
from einops import repeat

act2text = {
            'procgen':[
                "procgen game; LEFT DOWN direction",#0
                "procgen game; LEFT direction",#1 
                "procgen game; LEFT UP direction",#2
                "procgen game; DOWN direction",#3
                "procgen game; stays still and does not move",#4
                "procgen game; UP direction",#5
                "procgen game; RIGHT DOWN direction",#6
                "procgen game; RIGHT direction",#7
                "procgen game; RIGHT UP direction",#8
                "procgen game; action D",
                "procgen game; action A",
                "procgen game; action W",
                "procgen game; action S",
                "procgen game; action Q",
                "procgen game; action E",
                ],

            'Breakout': [
                "breakout game; action NOOP",
                "breakout game; action FIRE",
                "breakout game; action RIGHT",
                "breakout game; action LEFT",
                ],
            'Boxing': [
                "Atari boxing game; action NOOP",
                "Atari boxing game; action FIRE",
                "Atari boxing game; action UP",
                "Atari boxing game; action RIGHT",
                "Atari boxing game; action LEFT",
                "Atari boxing game; action DOWN",
                "Atari boxing game; action UPRIGHT",
                "Atari boxing game; action UPLEFT",
                "Atari boxing game; action DOWNRIGHT",
                "Atari boxing game; action DOWNLEFT",
                "Atari boxing game; action UPFIRE",
                "Atari boxing game; action RIGHTFIRE",
                "Atari boxing game; action LEFTFIRE",
                "Atari boxing game; action DOWNFIRE",
                "Atari boxing game; action UPRIGHTFIRE",
                "Atari boxing game; action UPLEFTFIRE",
                "Atari boxing game; action DOWNRIGHTFIRE",
                "Atari boxing game; action DOWNLEFTFIRE",
            ],
            'KungFuMaster': [
            "Atari KungFuMaster game; action NOOP.",
            "Atari KungFuMaster game; action UP.",
            "Atari KungFuMaster game; action RIGHT.",
            "Atari KungFuMaster game; action LEFT.",
            "Atari KungFuMaster game; action DOWN.",
            "Atari KungFuMaster game; action DOWNRIGHT.",
            "Atari KungFuMaster game; action DOWNLEFT.",
            "Atari KungFuMaster game; action RIGHTFIRE.",
            "Atari KungFuMaster game; action LEFTFIRE.",
            "Atari KungFuMaster game; action DOWNFIRE.",
            "Atari KungFuMaster game; action UPRIGHTFIRE.",
            "Atari KungFuMaster game; action UPLEFTFIRE.",
            "Atari KungFuMaster game; action DOWNRIGHTFIRE.",
            "Atari KungFuMaster game; action DOWNLEFTFIRE.",
            ],
        'MsPacman': [
            "Atari MsPacman game; action NOOP.",
            "Atari MsPacman game; action UP.",
            "Atari MsPacman game; action RIGHT.",
            "Atari MsPacman game; action LEFT.",
            "Atari MsPacman game; action DOWN.",
            "Atari MsPacman game; action UPRIGHT.",
            "Atari MsPacman game; action UPLEFT.",
            "Atari MsPacman game; action DOWNRIGHT.",
            "Atari MsPacman game; action DOWNLEFT.",
            ],
        'Alien':[
                "Atari Alien game; action NOOP",
                "Atari Alien game; action FIRE",
                "Atari Alien game; action UP",
                "Atari Alien game; action RIGHT",
                "Atari Alien game; action LEFT",
                "Atari Alien game; action DOWN",
                "Atari Alien game; action UPRIGHT",
                "Atari Alien game; action UPLEFT",
                "Atari Alien game; action DOWNRIGHT",
                "Atari Alien game; action DOWNLEFT",
                "Atari Alien game; action UPFIRE",
                "Atari Alien game; action RIGHTFIRE",
                "Atari Alien game; action LEFTFIRE",
                "Atari Alien game; action DOWNFIRE",
                "Atari Alien game; action UPRIGHTFIRE",
                "Atari Alien game; action UPLEFTFIRE",
                "Atari Alien game; action DOWNRIGHTFIRE",
                "Atari Alien game; action DOWNLEFTFIRE",
        ],
        'Amidar':[
                "Atari Amidar game; action NOOP",
                "Atari Amidar game; action FIRE",
                "Atari Amidar game; action UP",
                "Atari Amidar game; action RIGHT",
                "Atari Amidar game; action LEFT",
                "Atari Amidar game; action DOWN",
                "Atari Amidar game; action UPFIRE",
                "Atari Amidar game; action RIGHTFIRE",
                "Atari Amidar game; action LEFTFIRE",
                "Atari Amidar game; action DOWNFIRE",
                
        ],
        'Assault':[
             "Atari Assault game; action NOOP",
                "Atari Assault game; action FIRE",
                "Atari Assault game; action UP",
                "Atari Assault game; action RIGHT",
                "Atari Assault game; action LEFT",
                "Atari Assault game; action RIGHTFIRE",
                "Atari Assault game; action LEFTFIRE",
               
        ],
        'Asterix':[
             "Atari Asterix game; action NOOP",
                "Atari Asterix game; action UP",
                "Atari Asterix game; action RIGHT",
                "Atari Asterix game; action LEFT",
                "Atari Asterix game; action DOWN",
                "Atari Asterix game; action UPRIGHT",
                "Atari Asterix game; action UPLEFT",
                "Atari Asterix game; action DOWNRIGHT",
                "Atari Asterix game; action DOWNLEFT",
        ],
        'Asteroids':[
             "Atari Asteroids game; action NOOP",
                "Atari Asteroids game; action FIRE",
                "Atari Asteroids game; action UP",
                "Atari Asteroids game; action RIGHT",
                "Atari Asteroids game; action LEFT",
                "Atari Asteroids game; action DOWN",
                "Atari Asteroids game; action UPRIGHT",
                "Atari Asteroids game; action UPLEFT",
                "Atari Asteroids game; action UPFIRE",
                "Atari Asteroids game; action RIGHTFIRE",
                "Atari Asteroids game; action LEFTFIRE",
                "Atari Asteroids game; action DOWNFIRE",
                "Atari Asteroids game; action UPRIGHTFIRE",
                "Atari Asteroids game; action UPLEFTFIRE",
        ],
        'Atlantis':[
             "Atari Atlantis game; action NOOP",
                "Atari Atlantis game; action FIRE",
                "Atari Atlantis game; action RIGHTFIRE",
                "Atari Atlantis game; action LEFTFIRE",
        ],
        'BankHeist':[
             "Atari BankHeist game; action NOOP",
                "Atari BankHeist game; action FIRE",
                "Atari BankHeist game; action UP",
                "Atari BankHeist game; action RIGHT",
                "Atari BankHeist game; action LEFT",
                "Atari BankHeist game; action DOWN",
                "Atari BankHeist game; action UPRIGHT",
                "Atari BankHeist game; action UPLEFT",
                "Atari BankHeist game; action DOWNRIGHT",
                "Atari BankHeist game; action DOWNLEFT",
                "Atari BankHeist game; action UPFIRE",
                "Atari BankHeist game; action RIGHTFIRE",
                "Atari BankHeist game; action LEFTFIRE",
                "Atari BankHeist game; action DOWNFIRE",
                "Atari BankHeist game; action UPRIGHTFIRE",
                "Atari BankHeist game; action UPLEFTFIRE",
                "Atari BankHeist game; action DOWNRIGHTFIRE",
                "Atari BankHeist game; action DOWNLEFTFIRE",
        ],
        'BattleZone':[
             "Atari BattleZone game; action NOOP",
                "Atari BattleZone game; action FIRE",
                "Atari BattleZone game; action UP",
                "Atari BattleZone game; action RIGHT",
                "Atari BattleZone game; action LEFT",
                "Atari BattleZone game; action DOWN",
                "Atari BattleZone game; action UPRIGHT",
                "Atari BattleZone game; action UPLEFT",
                "Atari BattleZone game; action DOWNRIGHT",
                "Atari BattleZone game; action DOWNLEFT",
                "Atari BattleZone game; action UPFIRE",
                "Atari BattleZone game; action RIGHTFIRE",
                "Atari BattleZone game; action LEFTFIRE",
                "Atari BattleZone game; action DOWNFIRE",
                "Atari BattleZone game; action UPRIGHTFIRE",
                "Atari BattleZone game; action UPLEFTFIRE",
                "Atari BattleZone game; action DOWNRIGHTFIRE",
                "Atari BattleZone game; action DOWNLEFTFIRE",
        ],
        'BeamRider':[
             "Atari BeamRider game; action NOOP",
                "Atari BeamRider game; action FIRE",
                "Atari BeamRider game; action UP",
                "Atari BeamRider game; action RIGHT",
                "Atari BeamRider game; action LEFT",
                "Atari BeamRider game; action UPRIGHT",
                "Atari BeamRider game; action UPLEFT",
                "Atari BeamRider game; action RIGHTFIRE",
                "Atari BeamRider game; action LEFTFIRE",
        ],
        'Bowling':[
             "Atari Bowling game; action NOOP",
                "Atari Bowling game; action FIRE",
                "Atari Bowling game; action UP",
                "Atari Bowling game; action DOWN",
                "Atari Bowling game; action UPFIRE",
                "Atari Bowling game; action DOWNFIRE",
        ]
    }

def parse_prompt(action_in_id):
    prompt='This is a video of computer game, excecuting actions'
    for action_id in action_in_id:
        prompt+=', '+act2text['procgen'][action_id][14:]
    return prompt

def preprocess_image(image, torch_dtype=torch.bfloat16, device='cpu', pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype , device=device )
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


def preprocess_video(video, torch_dtype=torch.bfloat16, device='cpu', pattern="B C T H W", min_value=-1, max_value=1):
    # Transform a list of PIL.Image to torch.Tensor
    video = [preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
    video = torch.stack(video, dim=pattern.index("T") // 2).squeeze(0)
    return video

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split, height, width, num_frames, sample_num=None):
        assert split in ['same', 'iid', 'ood']

        self.base_path = base_path
        self.video_path=os.path.join(base_path, 'video')
        self.video_names=os.listdir(self.video_path)
        self.video_names=[x for x in self.video_names if x.endswith('.mp4')]
        self.video_names=sorted(self.video_names)

        self.info=np.load(os.path.join(base_path, 'info.npy'), allow_pickle=True).item()
        self.split=split
        self.height=height
        self.width=width
        self.num_frames=num_frames
        self.same_items=self.info['same_items']
        self.iid_items=self.info['iid_items']
        self.ood_items=self.info['ood_items']
        self.action_dataset=np.load(os.path.join(base_path, 'action.npy'))

        self.sample_num=None
        self.repeat=1
        if sample_num is not None:
            assert sample_num<=100 and 100%sample_num==0
            self.same_items=self.same_items[:sample_num]
            self.iid_items=self.iid_items[:sample_num]
            self.ood_items=self.ood_items[:sample_num]
            self.sample_num=sample_num
            self.repeat=100//sample_num
    
    def __len__(self):
        if self.split == 'same':
            return len(self.same_items)*self.repeat
        elif self.split == 'iid':
            return len(self.iid_items)*self.repeat
        elif self.split == 'ood':
            return len(self.ood_items)*self.repeat
    
    def __getitem__(self, idx):
        if self.sample_num is not None:
            idx=idx%self.sample_num
        
        if self.split == 'same':
            video_idx,start_idx=self.same_items[idx]
        elif self.split == 'iid':
            video_idx,start_idx=self.iid_items[idx]
        elif self.split == 'ood':
            video_idx,start_idx=self.ood_items[idx]
        
        video_dir=os.path.join(self.video_path, self.video_names[video_idx])
        raw_video=VideoData(video_dir, height=self.height, width=self.width)
        raw_video=[raw_video[i] for i in range(start_idx,start_idx+self.num_frames)]
        video=preprocess_video(raw_video)                                       # [C,T,H,W] (-1,1)
        action=self.action_dataset[video_idx,start_idx:start_idx+self.num_frames-1] # [T-1,]
        # prompt=parse_prompt(action)
        return {
            'video':video,
            'prompt':action
        }