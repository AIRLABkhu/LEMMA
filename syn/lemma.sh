#for cifar
python recover_cifar.py \
--arch-name "resnet32x4" \
--exp-name "cifar100_rn32x4_1K" \
--batch-size 100 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 0

python recover_cifar.py \
--arch-name "vgg13" \
--exp-name "cifar100_vgg13_1K" \
--batch-size 100 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 0

python recover_cifar.py \
--arch-name "ResNet50" \
--exp-name "cifar100_RN50_1K" \
--batch-size 100 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 1

python recover_cifar.py \
--arch-name "resnet56" \
--exp-name "cifar100_rn56_1K" \
--batch-size 100 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 2

python recover_cifar.py \
--arch-name "resnet110" \
--exp-name "cifar100_rn110_1K" \
--batch-size 100 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 3



python recover_cifar.py \
--arch-name "wrn_40_2" \
--exp-name "cifar100_wrn40_1K" \
--batch-size 100 \
--lr 0.25 \
--iteration 1000 \
--r-bn 0.01 \
--store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 4

# for imagenet

python data_synthesis.py \
--arch-name "ResNet34" \
--exp-name "imagnet1k_RN34" \
--batch-size 100 \
--lr 0.25 \
--iteration 4000 \
--l2-scale 0 --tv-l2 0 --r-bn 0.01 \
--verifier --store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 5

python data_synthesis.py \
--arch-name "ResNet50" \
--exp-name "imagnet1k_RN50" \
--batch-size 100 \
--lr 0.25 \
--iteration 4000 \
--l2-scale 0 --tv-l2 0 --r-bn 0.01 \
--verifier --store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 7

python data_synthesis.py \
--arch-name "ResNet18" \
--exp-name "imagnet1k_RN18" \
--batch-size 100 \
--lr 0.25 \
--iteration 4000 \
--l2-scale 0 --tv-l2 0 --r-bn 0.01 \
--verifier --store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 2

python data_synthesis.py \
--arch-name "ResNet34" \
--exp-name "imagnet1k_RN34" \
--batch-size 100 \
--lr 0.25 \
--iteration 4000 \
--l2-scale 0 --tv-l2 0 --r-bn 0.01 \
--verifier --store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 0

python data_synthesis.py \
--arch-name "ResNet50" \
--exp-name "imagnet1k_RN50" \
--batch-size 100 \
--lr 0.25 \
--iteration 4000 \
--l2-scale 0 --tv-l2 0 --r-bn 0.01 \
--verifier --store-best-images \
--ipc-start 0 --ipc-end 10 \
--device 1
