TORCH_DISTRIBUTED_DEBUG=DETAIL BASICSR_JIT=True CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master-port 12345 --nproc_per_node 4 basicsr/all_in_one_train.py -opt options/all_in_one/pretrain/pretrain_NAFNet_AIO_10d.yml --launcher pytorch
TORCH_DISTRIBUTED_DEBUG=DETAIL BASICSR_JIT=True CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master-port 12345 --nproc_per_node 4 basicsr/all_in_one_train.py -opt options/all_in_one/train/train_NAFNet_AIO_10d.yml --launcher pytorch