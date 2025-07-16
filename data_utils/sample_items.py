import os
import numpy as np

def sample(base_path, game, num_frames, test_num):
    data_folder=os.path.join(base_path, game)
    episode_names=os.listdir(data_folder)
    episode_names=[x for x in episode_names if x.startswith('episode_')]
    episode_num=len(episode_names)-test_num

    items=[]
    for i in range(episode_num):
        episode=np.load(os.path.join(data_folder, f'episode_{i}.npz'))
        episode_length=episode['action'].shape[0]
        clip_num=episode_length-num_frames+1

        if clip_num<=0:
            continue

        episode_idx=np.ones(clip_num,dtype=np.int32)*i
        episode_idx=episode_idx.reshape(-1,1)

        step_idx=np.arange(clip_num,dtype=np.int32).reshape(-1,1)

        episode_items=np.concatenate([episode_idx, step_idx],axis=1)
        items.append(episode_items)

    items=np.concatenate(items,axis=0)
    print(f"{game} Total number of clips: {len(items)}")
    np.savez(os.path.join(data_folder, f'info_new.npz'), items=items)

    # test
    test_items=[]
    for i in range(episode_num,episode_num+test_num):
        episode=np.load(os.path.join(data_folder, f'episode_{i}.npz'))
        episode_length=episode['action'].shape[0]
        clip_num=episode_length-num_frames+1
        
        if clip_num<=0:
            continue
        
        episode_idx=np.ones(clip_num,dtype=np.int32)*i
        episode_idx=episode_idx.reshape(-1,1)
        
        step_idx=np.arange(clip_num,dtype=np.int32).reshape(-1,1)

        episode_items=np.concatenate([episode_idx, step_idx],axis=1)
        test_items.append(episode_items)
    
    test_items=np.concatenate(test_items,axis=0)
    # sample 256 items
    test_items=test_items[np.random.choice(len(test_items), size=256, replace=False)]
    print(f"{game} Total number of test clips: {len(test_items)}")
    np.savez(os.path.join(data_folder, f'info_new_test.npz'), items=test_items)

if __name__ == "__main__":
    sample(base_path='./../data/procgen',
           game='ninja',
           num_frames=9,
           test_num=10)
    sample(base_path='./../data/procgen',
           game='coinrun',
           num_frames=9,
           test_num=10)
    sample(base_path='./../data/procgen',
           game='jumper',
           num_frames=9,
           test_num=10)