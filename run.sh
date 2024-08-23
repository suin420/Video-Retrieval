python inference.py --datetime=./data/hollywood2   --arch=clip_stochastic \
  --videos_dir=./data/hollywood2/test_clips  --batch_size=16 --noclip_lr=3e-5 \
  --transformer_dropout=0.3  --dataset_name=hw2   --stochasic_trials=20 --gpu='0' \
   --load_epoch=0  --num_epochs=5  --exp_name=hw2
