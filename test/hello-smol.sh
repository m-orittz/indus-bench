
# Launch finetuning on your dataset
python3 lerobot/scripts/train.py \
  --policy.path=n3puiol/smolvla_pusht \
  --dataset.repo_id=lerobot/svla_so101_pickplace \ 
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true

# Evaluate n3puiol's smolvla_pusht, 
# Video output: ~/outputs/eval/2025-08-12/16-42-12_pusht_smolvla/videos
lerobot-eval \
  --policy.path=n3puiol/smolvla_pusht \
  --env.type=pusht \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --policy.use_amp=false \
  --policy.device=cuda
