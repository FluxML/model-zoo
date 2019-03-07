# MemN2N

This model implements training of language model on Penn Treebank (PTB) dataset using End-To-End Memory Networks

## Usage
    julia main.jl --edim 20 --lindim 0 --nhops 1 --init_hid 0.1 --init_std 0.01 --batch_size 8 --mem_size 40 --epochs 1 --init_lr 0.01 --max_grad_norm 50 "/path/to/penn_tree_bank/"
