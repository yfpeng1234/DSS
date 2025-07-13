import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import VideoDataset
import lpips
import piqa

def load_model(ckpt_dir):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
    )
    pipe.load_lora(pipe.dit, ckpt_dir, alpha=1)
    # pipe.enable_vram_management()
    return pipe

def load_dataset():
    dataset=VideoDataset(
        base_path="data/sample",
        metadata_path="data/sample/metadata.csv",
        height=256,
        width=256,
        num_frames=9,
        repeat=100,
    )
    return dataset

def video_predict(pipe, prompt,):
    video = pipe(
        prompt=prompt,
        seed=1,
        tiled=False,
        height=256,
        width=256,
        num_frames=9,
        cfg_scale=1.0,
    )
    return video

class PerceptualMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=torch.nn.MSELoss()
        self.lpips=lpips.LPIPS(net='vgg')
        self.psnr=piqa.PSNR(epsilon=1e-08, value_range=1.0)
        self.ssim=piqa.SSIM()

    @torch.no_grad()
    def forward(self,gt,pred):
        '''
        make sure that input is [T,3,H,W] [0,1]
        '''
        mse=self.mse(gt,pred).item()
        lpips=self.lpips(gt,pred).mean().item()
        psnr=self.psnr(gt,pred).item()
        ssim=self.ssim(gt,pred).item()
        return mse,lpips,psnr,ssim
    
def save_video_as_image(video1,video2,video3,image_path):
    '''
    require video [T,H,W,C] (0,255)
    '''
    T, H, W, C = video1.shape
    video1=video1.to(device='cpu',dtype=torch.uint8)
    video2=video2.to(device='cpu',dtype=torch.uint8)
    video3=video3.to(device='cpu',dtype=torch.uint8)

    rows=[]
    rows.append(video1.permute(1, 0, 2, 3).reshape(H, T * W, C))
    rows.append(video2.permute(1, 0, 2, 3).reshape(H, T * W, C))
    rows.append(video3.permute(1, 0, 2, 3).reshape(H, T * W, C))
    grid=torch.cat(rows,dim=0)

    img=Image.fromarray(grid.numpy())
    img.save(image_path)

if __name__ == "__main__":
    # load model
    ckpt_dir = "ckpt/demo_lora/epoch-39.safetensors"
    pipe = load_model(ckpt_dir)

    # load dataset
    # import ipdb; ipdb.set_trace()
    dataset = load_dataset()
    data=dataset[0]
    video=data["video"]
    prompt=data["prompt"]

    # metric
    metrics = PerceptualMetrics().to(device=pipe.device, dtype=pipe.torch_dtype)

    # gt and upper bound
    input_video=pipe.preprocess_video(video)        # [1,3,9,256,256] (-1,1)
    input_latents=pipe.vae.encode(input_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device) # [1,16,3,32,32]
    decoded_video=pipe.vae.decode(input_latents, device=pipe.device) # [1,3,9,256,256] (-1,1)

    # video prediction
    pred_video=video_predict(pipe, prompt)      # [1,3,9,256,256] (-1,1)

    # calculate metrics
    gt=input_video.squeeze(0).permute(1,0,2,3)*0.5 + 0.5
    ub=decoded_video.squeeze(0).permute(1,0,2,3)*0.5 + 0.5
    pred=pred_video.squeeze(0).permute(1,0,2,3)*0.5 + 0.5
    mse, lpips, psnr, ssim = metrics(gt, pred)
    mse_ub, lpips_ub, psnr_ub, ssim_ub = metrics(gt, ub)
    print(f"Metrics for prediction: MSE={mse}, LPIPS={lpips}, PSNR={psnr}, SSIM={ssim}")
    print(f"Metrics for upper bound: MSE={mse_ub}, LPIPS={lpips_ub}, PSNR={psnr_ub}, SSIM={ssim_ub}")

    # save video
    video1=input_video.cpu().squeeze(0).permute(1,2,3,0)*127.5 + 127.5
    video2=decoded_video.cpu().squeeze(0).permute(1,2,3,0)*127.5 + 127.5
    video3=pred_video.cpu().squeeze(0).permute(1,2,3,0)*127.5 + 127.5
    os.makedirs("result/demo_lora", exist_ok=True)
    save_video_as_image(video1, video2, video3, "result/demo_lora/demo_lora_result.png")