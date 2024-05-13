'''
# python tools/extract.py --device cuda --model resnet56   --all
# python tools/extract.py --device cuda --model resnet110  --all
# python tools/extract.py --device cuda --model resnet32x4 --all
# python tools/extract.py --device cuda --model wrn_40_2   --all
# python tools/extract.py --device cuda --model vgg13      --all
# python tools/extract.py --device cuda --model ResNet50   --lightweight
python tools/extract.py --device cuda --model ResNet50   --feats
python tools/extract.py --device cuda --model ResNet50   --preact-feats
'''

import os
import argparse
from tqdm.auto import tqdm

import torch
from torch import nn
from mdistiller import dataset, models

teacher_names = filter(lambda x: x.endswith('_mem'), models.cifar_model_dict.keys())
teacher_names = map(lambda x: x.replace('_mem', ''), teacher_names)
teacher_names = list(teacher_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--model', type=str, default='resnet56', choices=teacher_names)
    parser.add_argument('--logits', default=False, action='store_true')
    parser.add_argument('--feats', default=False, action='store_true')
    parser.add_argument('--preact-feats', default=False, action='store_true')
    parser.add_argument('--pooled-feat', default=False, action='store_true')
    parser.add_argument('--lightweight', default=False, action='store_true')
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    
    get_logits, get_feats, get_preact_feats, get_pooled_feat = False, False, False, False
    if args.all:
        get_logits, get_feats, get_preact_feats, get_pooled_feat = True, True, True, True
    elif args.lightweight:
        get_logits, get_feats, get_preact_feats, get_pooled_feat = True, False, False, True
    if args.logits: get_logits = True
    if args.feats: get_feats = True
    if args.preact_feats: get_preact_feats = True
    if args.pooled_feat: get_pooled_feat = True
    
    if not any([get_logits, get_feats, get_preact_feats, get_pooled_feat]):
        print('Please specify the data to memorize.')
        exit()

    device = args.device
    train_loader, _, _ = dataset.get_cifar100_dataloaders(args.batch_size, args.batch_size, 2, train_like_test=True)
    model_type, state_dict_path = models.cifar_model_dict[args.model]
    state_dict = torch.load(state_dict_path, map_location='cpu')['model']
    _, memory_dir = models.cifar_model_dict[f'{args.model}_mem']
    memory_dir = os.path.abspath(memory_dir)
    logits_filename = os.path.join(memory_dir, 'logits.pth')
    feats_filename = os.path.join(memory_dir, 'feats.pth')
    preact_feats_filename = os.path.join(memory_dir, 'preact_feats.pth')
    pooled_feat_filename = os.path.join(memory_dir, 'pooled_feat.pth')
    
    if not os.path.exists(memory_dir):
        os.mkdir(memory_dir)

    model = model_type(num_classes=100).to(device).train()
    model.load_state_dict(state_dict)
    if device != 'cpu':
        model = nn.DataParallel(model)
    
    with torch.no_grad():
        all_logits, all_feats, all_preact_feats, all_pooled_feat = [], [], [], []
        for input, _, _ in tqdm(train_loader, desc=args.model, dynamic_ncols=True):
            input = input.to(device)
            logits, features = model(input)
            
            if get_logits: all_logits.append(logits.cpu())
            if get_feats: all_feats.append(list(map(torch.Tensor.cpu, features['feats'])))
            if get_preact_feats: all_preact_feats.append(list(map(torch.Tensor.cpu, features['preact_feats'])))
            if get_pooled_feat: all_pooled_feat.append(features['pooled_feat'].cpu())
    
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
    if get_logits:
        logits = torch.cat(all_logits)
        save_fn(logits, logits_filename, 'Logits')
        del logits
    if get_feats:
        feats = list(map(torch.cat, zip(*all_feats)))
        save_fn(feats, feats_filename, 'Feats')
        del feats
    if get_preact_feats:
        preact_feats = list(map(torch.cat, zip(*all_feats)))
        save_fn(preact_feats, preact_feats_filename, 'Preact Feats')
        del preact_feats
    if get_pooled_feat:
        pooled_feat = torch.cat(all_pooled_feat)
        save_fn(pooled_feat, pooled_feat_filename, 'Pooled Feat')
        del pooled_feat
    