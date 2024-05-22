from typing import Iterable
import numpy as np


def read(filename: str, step_size=9):
    with open(filename, 'r') as file:
        lines = list(map(str.strip, file.readlines()))
    for i in range(step_size - 2):
        yield tuple(map(lambda x: float(x.split(': ')[-1]), lines[i+1::step_size]))

def crawl(pattern, args, sort=True):
    args = [arg if isinstance(arg, Iterable) else [arg] for arg in args]
    filenames = [*map(lambda x: pattern.format(*x), args)]
    if sort:
        filenames = sorted(filenames)
        
    logs = [*map(lambda x: tuple(read(x)), filenames)]
    results = []
    for i in range(7):
        item = [log[i] for log in logs]
        min_len = min(map(len, item))
        item = np.array([*map(lambda x: x[:min_len], item)])
        results.append(item)
    return dict(zip([
        'epoch',
        'lr',
        'train_top1',
        'train_loss',
        'valid_top1',
        'valid_top5',
        'valid_loss',
    ], results))
    
def smooth(array, weight: float=0.667):
    last = 0
    weight_ = 1 - weight
    result = np.array(array)
    for i in range(len(result)):
        result[i] = weight * last + weight_ * array[i]
        last = result[i]
    return result
        

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--pattern', type=str, required=True)
    parser.add_argument('--args', type=eval, required=True)
    args = parser.parse_args()
    
    output = crawl(args.pattern, args.args)
    for key, val in output.items():
        print(key, ':', val.shape)
