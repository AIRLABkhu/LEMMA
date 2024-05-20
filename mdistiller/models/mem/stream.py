import os
from pathlib import Path
import multiprocessing as mp
import numpy as np
import shutil


class MemStream:
    '''
    `mem_name`: The directory in the `mem_dir` that contains subdirectories by memory types. 
    `num_workers`: The number of processes. 
    `log_dir`: The log directory where the clone memory directory will be located. 
    `mem_dir`: The base directory of the original memory. 
    '''
    def __init__(self, mem_name: str, num_workers: int, log_dir: str, mem_dir: str=None):
        if mem_dir is None:
            mem_dir = Path(__file__).parents[4]
        else:
            mem_dir = Path(mem_dir)
        
        self.mem_dir = Path(mem_name).expanduser().resolve().joinpath(mem_name)
        self.log_dir = Path(log_dir).expanduser().resolve().joinpath('memory')
        self.num_workers = num_workers
        self.worker_pool = mp.Pool(processes=num_workers)
        
        self.mem_filename_fmt = f'{str(self.mem_dir)}{os.sep}{{:s}}'.format
        self.log_filename_fmt = f'{str(self.log_dir)}{os.sep}{{:s}}'.format
        
        if not self.mem_dir.exists():
            raise FileNotFoundError(f'Memory not found "{str(self.mem_dir)}".')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def read(self, filenames):
        log_fn = map(self.log_filename_fmt, filenames)
        mem_fn = map(self.mem_filename_fmt, filenames)
        return self.worker_pool.map(MemStream.__read_fn, zip(log_fn, mem_fn))
        
    def write(self, filenames, data):
        log_fn = map(self.log_filename_fmt, filenames)
        self.worker_pool.map(MemStream.__write_fn, zip(log_fn, data))
    
    def __read_fn(bundle):
        log_fn, mem_fn = bundle
        if not os.path.exists(log_fn):
            shutil.copy(mem_fn, log_fn)
        return np.load(log_fn)
    
    def __write_fn(bundle):
        log_fn, data = bundle
        np.save(log_fn, data)
