export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/home/image_gen/model_op/"
# export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="/home/image_gen/train/"

accelerate launch --mixed_precision="bf16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="plot" \
  --seed=1337
