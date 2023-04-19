from Gen_Baseline.Solver_Gen import Solver_Gen
from options import SharinOptions
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
import heapq 
from tqdm import tqdm
import os


def main_worker(gpu, ngpus_per_node, opt):
    opt.gpu = gpu
    
    if opt.gpu is not None:
        print("=> Use GPU: {} for training".format(opt.gpu))
        
    
    opt.rank = opt.gpu # since we only have one node
    if opt.distributed:
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
    
    solver = Solver_Gen(opt)
    solver.load_prev_model()
    
    START_ITER = solver.START_ITER
    
    # used to enable tqdm progress display
    if not opt.hpc and opt.rank % opt.ngpus == 0:
        pbar = tqdm(total=opt.total_iterations-START_ITER) 

    for iter_id in range(START_ITER, opt.total_iterations): 
        solver.train_iter(iter_id)
        
        if iter_id % 1000 == 999:
        # if iter_id % 10 == 9:
            if opt.rank % opt.ngpus == 0:
                print("iteration finished: {}/{}".format(iter_id, opt.total_iterations - START_ITER + 1))
                
                solver.save_model(iter_id)
                
        # used to enable tqdm progress display
        if not opt.hpc and opt.rank % opt.ngpus == 0:
            pbar.update(1)
    
    # close the program
    if not opt.hpc and opt.rank % opt.ngpus == 0:
        pbar.close()
    
    if opt.rank % opt.ngpus == 0:
        solver.writer.close()
    
    if opt.distributed:
        dist.destroy_process_group()

if __name__=='__main__':
    options = SharinOptions()
    opt = options.parse()
    
    
    opt.distributed = True if opt.ngpus > 1 else False
    if opt.distributed and opt.dist_url == "env://":
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
    if opt.ngpus > torch.cuda.device_count():
        raise ValueError("Not enough gpus!")
    
    
    # NOTE: assume we have one node with # ngpus of processes, each using one gpu
    if opt.distributed:
        opt.world_size = opt.ngpus 
        mp.spawn(main_worker, nprocs=opt.ngpus, args=(opt.ngpus, opt))
    else:
        assert opt.ngpus == 1
        main_worker(0, 1, opt)
    
