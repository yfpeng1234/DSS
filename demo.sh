export CUDA_VISIBLE_DEVICES=2,3,4,5
accelerate launch --num_processes 4 examples/wanvideo/model_training/train.py \
  --dataset_base_path data/sample \
  --dataset_metadata_path data/sample/metadata.csv \
  --height 256 \
  --width 256 \
  --num_frames 9 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 400 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./ckpt/baseline_lora_v4" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32