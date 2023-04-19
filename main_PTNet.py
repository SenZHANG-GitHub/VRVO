from PTNet_Baseline.Solver_PTNet import Solver_PTNet
from PTNet_Baseline.Validater_PTNet import Validater_PTNet
from options import SharinOptions
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
import heapq 
from tqdm import tqdm
import os

# NOTE: Remain to do: only maintain the best k models
# use heapq (min heap) (heapify, heappush, heappop) 
# if min heap item is tuple (i1, i2), will use the first value (i1) for cmp
# https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/
top_k_val = []
heapq.heapify(top_k_val)


def main_worker(gpu, ngpus_per_node, opt):
    global top_k_val
    opt.gpu = gpu
    
    if opt.gpu is not None:
        print("=> Use GPU: {} for training".format(opt.gpu))
        
    
    opt.rank = opt.gpu # since we only have one node
    if opt.distributed:
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
    
    solver = Solver_PTNet(opt)
    solver.load_prev_model()
    
    START_ITER = solver.START_ITER
    
    # used to enable tqdm progress display
    if not opt.hpc and opt.rank % opt.ngpus == 0:
        pbar = tqdm(total=opt.total_iterations-START_ITER) 
    
    for iter_id in range(START_ITER, opt.total_iterations):
        
        solver.train_iter(iter_id)
        
        if iter_id % 1000 == 999:
        # if iter_id % 10 == 9:
        
            # print("====================================================")
            # print("=> gpu {}: iteration finished: {}/{}, sub_batch_size: {} ".format(opt.gpu, iter_id, opt.total_iterations - START_ITER, solver.batch_size))
            # val_loss = solver.val(iter_id).cpu().item()
            
            if opt.rank % opt.ngpus == 0:
                print("====================================================")
                print("=> gpu {}: iteration finished: {}/{}, sub_batch_size: {} ".format(opt.gpu, iter_id, opt.total_iterations - START_ITER, solver.batch_size))
                
                val_loss = solver.val(iter_id).cpu().item()
                
                if len(top_k_val) < opt.top_k_val:
                    # -1* because heapq is min heap rather than max heap
                    heapq.heappush(top_k_val, (-1*val_loss, iter_id))
                    solver.save_model(iter_id)
                else:
                    border_val_loss, border_iter = heapq.heappop(top_k_val)
                    border_val_loss *= -1
                    if val_loss < border_val_loss:
                        heapq.heappush(top_k_val, (-1*val_loss, iter_id))
                        solver.rm_model(border_iter)
                        solver.save_model(iter_id)
                    else:
                        heapq.heappush(top_k_val, (-1*border_val_loss, border_iter))
                        
                print("=> current best models (-1*val_loss, iteration): \n{}".format(top_k_val))
        
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


def preprocess_opts(opt):
    """Setup opt for validation
    => If opt.val is True: only VAL_OPTS will be maintained
    => Other options are changed to the saved opts of the exp
    """
    if not opt.val:
        return opt
    
    VAL_OPTS = ["batch_size", "post_process", "val", "val_average_disp", "val_depth_mode"]

    remain_opts = {}
    for key in VAL_OPTS:
        remain_opts[key] = opt.__dict__[key]
    
    opt_path = os.path.join("./PTNet_Baseline/tensorboard_logs/vKitti2", opt.exp, "PTNet_Baseline_bicubic/options/opt.json")
    assert os.path.isfile(opt_path), "opt.json for --exp {} does not exists".format(opt.exp)
    
    prev_model_path = os.path.join("./PTNet_Baseline/saved_models", opt.exp)
    assert os.path.isdir(prev_model_path), "saved_models/ for --exp {} does not exists".format(opt.exp)
    
    with open(opt_path, "r") as f:
        import json 
        prev_opt = json.load(f) 
    
    opt.__dict__.update(prev_opt)
    for key in VAL_OPTS:
        opt.__dict__[key] = remain_opts[key]
    opt.__dict__["prev_model_path"] = prev_model_path
    opt.__dict__["ngpus"] = 1
    opt.__dict__["frame_ids"] = [0]
    
    return opt
        

def validate(opt):
    opt.gpu = 0
    opt.rank = 0
    
    validater = Validater_PTNet(opt)
    validater.validate()


if __name__=='__main__':
    options = SharinOptions()
    opt = options.parse()
    
    opt = preprocess_opts(opt)
    
    opt.distributed = True if opt.ngpus > 1 else False
    if opt.distributed and opt.dist_url == "env://":
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
    if opt.ngpus > torch.cuda.device_count():
        raise ValueError("Not enough gpus!")
    
    # NOTE: assume we have one node with # ngpus of processes, each using one gpu
    if opt.val:
        validate(opt)
    else:
        if opt.distributed:
            opt.world_size = opt.ngpus 
            mp.spawn(main_worker, nprocs=opt.ngpus, args=(opt.ngpus, opt))
        else:
            assert opt.ngpus == 1
            main_worker(0, 1, opt)
    
    

