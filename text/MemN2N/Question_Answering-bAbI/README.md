# MemN2N

Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

## Usage
    julia main.jl --edim 20 --lindim 0 --nhops 1 --init_hid 0.1 --init_std 0.01 --batch_size 8 --mem_size 40 --epochs 1 --init_lr 0.01 --max_grad_norm 50 "/path/to/tasks_1-20_v1-2/en/"
