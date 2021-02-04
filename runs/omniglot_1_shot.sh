python main.py \
  --savedir './results/omni_1shot' \
  --dataset 'omniglot' \
  --mode 'train' \
  --batch_size 8 \
  --num_adapt_steps 5 \
  --inner_lr 0.1 \
  --num_ways 20 \
  --num_shots 1 \
  --num_query 15 \
  --num_iters 40000 \
  --num_workers 4 \
  --meta_lr 1e-3 \
  --mc_steps 1 \
  --grad_clip 3 \

python main.py \
  --savedir './results/omni_1shot' \
  --dataset 'omniglot' \
  --mode 'test' \
  --batch_size 8 \
  --num_adapt_steps 5 \
  --inner_lr 0.1 \
  --num_ways 20 \
  --num_shots 1 \
  --num_query 15 \
  --num_test_iters 1000\
  --num_workers 4 \
  --meta_lr 1e-3 \
  --mc_steps 30