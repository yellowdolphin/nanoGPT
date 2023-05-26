import time

out_dir = '/kaggle/working'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'lovecraft'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'lovecraft'
init_from = 'gpt2-medium' # gpt2-xl is the largest GPT-2 model, 
                          # but even gpt2-large is too large for single P100!

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# lovecraft has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 25

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
