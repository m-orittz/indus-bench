
# Launch finetuning on your dataset
python3 lerobot/src/lerobot/scripts/train.py \
  --policy.path=n3puiol/smolvla_pusht \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/pusht-train-test2 \
  --job_name=pusht-train-test2 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
  #If you do not want to push your model to the hub after training use:
  # --policy.push_to_hub=false

# BIG TRAIN
python3 lerobot/src/lerobot/scripts/train.py \
  --policy.path=n3puiol/smolvla_pusht \
  --dataset.repo_id=lerobot/pusht \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/pusht-train-test2 \
  --job_name=pusht-train-test2 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false

# Evaluate n3puiol's smolvla_pusht, 
# Video output: ~/outputs/eval/2025-08-12/16-42-12_pusht_smolvla/videos
lerobot-eval \
  --policy.path=n3puiol/smolvla_pusht \
  --env.type=pusht \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --policy.use_amp=false \
  --policy.device=cuda
