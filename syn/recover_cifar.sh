CUDA_VISIBLE_DEVICES=0 \
python recover_cifar.py \
--arch-name "resnet32x4" \
--exp-name "cifar100_rn18_1K_mobile.lr0.25.bn0.01" \
--batch-size 100 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 50

