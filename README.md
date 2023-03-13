# Ens_NAS_OOD

## Usage
#### Start training Ensemble OFA super-net
```
CUDA_VISIBLE_DEVICES=1 python train_ofa_net.py \
--task 'expand' --phase 2 --ens 2 --lr 0.005 \
--e '0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0,1.25,1.5,1.75,2.0' \
--save-path 'exp/test'
``` 
#### start evolutionary search 
```
CUDA_VISIBLE_DEVICES=5 python ens_evo_search.py \
--ens 2 --expand_list '0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0,1.25,1.5,1.75,2.0' \
--pretrained 'exp/0218/ID50_Ens2_cecb_e[0.25~2.0]_w[2]_d[0,1]/checkpoint/model_best.pth.tar'
```

#### fine-tune searched sub-net
```
CUDA_VISIBLE_DEVICES=5 python finetune_cifar.py --manualSeed 1 \
--pretrained 'exp/0218/ID50_Ens2_cecb_e[0.25~2.0]_w[2]_d[0,1]/checkpoint/model_best.pth.tar' \
--beta 0.9999 --gama 1.0 \
--lr 0.1 --train-batch 512 --ens 2 --epochs 300 \
-c "checkpoint/0224/ID56_Searched_subnet2_ft300" 
```
## History
#### 03-13 
channel selection(mask method)
```
imp_est.py
```
The script is a demo of implementing mask channel selection for ensemble model with weight sharing.

On working :
 ```
 ofa/imagenet_classification/elastic_nn/modules/dynamic_layers.py & dynamic_op.py
    -DynamicMaskConvLayer()

ofa/imagenet_classification/elastic_nn/networks/ofa_resnets.py
    -OFAMskedResNets18()
 ```