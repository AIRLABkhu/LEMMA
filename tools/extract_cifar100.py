'''
# python tools/extract_cifar100.py --model resnet56 
# python tools/extract_cifar100.py --model resnet110 
# python tools/extract_cifar100.py --model resnet32x4 
# python tools/extract_cifar100.py --model wrn_40_2 
# python tools/extract_cifar100.py --model vgg13  
# python tools/extract_cifar100.py --model ResNet50 
'''

import os
import argparse
from tqdm.auto import tqdm

import torch
from torch import nn
from mdistiller import dataset, models
import h5py

teacher_names = filter(lambda x: x.endswith('_mem'), models.cifar_model_dict.keys())
teacher_names = map(lambda x: x.replace('_mem', ''), teacher_names)
teacher_names = list(teacher_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet56', choices=teacher_names)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    
    device = args.device
    train_loader, _, _ = dataset.get_cifar100_dataloaders(args.batch_size, args.batch_size, 2, train_like_test=True)
    model_type, state_dict_path = models.cifar_model_dict[args.model]
    state_dict = torch.load(state_dict_path, map_location='cpu')['model']
    _, memory_dir = models.cifar_model_dict[f'{args.model}_mem']
    
    memory_dir = os.path.abspath(memory_dir)
    memory_filename = os.path.join(memory_dir, 'memory.hdf5') 
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir, exist_ok=True)
    print(memory_filename)

    model = model_type(num_classes=100).to(device).train()
    model.load_state_dict(state_dict)
    if device != 'cpu':
        model = nn.DataParallel(model)
    
    with torch.no_grad():
        all_logits, all_pooled_feat = [], [] 
        for input, _, _ in tqdm(train_loader, desc=args.model, dynamic_ncols=True):
            input = input.to(device)
            logits, features = model(input)
            
            all_logits.append(logits.cpu())
            all_pooled_feat.append(features['pooled_feat'].cpu())
    
    def convert_size(size_in_bytes):
        import math
        if size_in_bytes == 0:
            return 0, 'B'
        
        units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_in_bytes, 1024)))
        
        return size_in_bytes / math.pow(1024, i), units[i]
    
    def save_fn(data, filename, tag):
        print(f'\t{tag}:')
        
        if isinstance(data, list):
            print(f'\t\tShape:')
            for i, d in enumerate(data):
                print(f'\t\t\t[{i}]: {list(d.shape)} (numel: {d.numel():,})')
        else:
            print(f'\t\tShape: {list(data.shape)} (numel: {data.numel():,})')
            
        print(f"\t\tPath: '{filename}'.")
        torch.save(data, filename)
        
        file_size = os.path.getsize(filename)
        short_size, unit = convert_size(file_size)
        print(f"\t\tSize: {short_size:.4f} {unit} ({file_size:,} B).")
    
    print('Processing...')
    print(f"\tMemory directory: '{memory_dir}'.")

    logits = torch.cat(all_logits).numpy()
    pooled_feat = torch.cat(all_pooled_feat).numpy()
    with h5py.File(memory_filename, 'w') as file:
        file.create_dataset('logits', data=logits)
        file.create_dataset('pooled_feat', data=pooled_feat)
    