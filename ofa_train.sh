CUDA_VISIBLE_DEVICES=1 python train_ofa_net.py \
--task 'expand' --phase 2 --ens 2 --lr 0.005 \
--e '0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0,1.25,1.5,1.75,2.0' \
--save-path 'exp/0222/ID55_reorganize'