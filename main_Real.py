from Real_Baseline.Solver_Real import Solver_Real
from options import SharinOptions
import os
import json
import torch
from tqdm import tqdm
import heapq


if __name__=='__main__':    
    options = SharinOptions()
    opt = options.parse()
        
    opt.gpu = 0
    opt.rank = 0
    opt.distributed = False
    
    solver = Solver_Real(opt)
        
    # NOTE: Remain to do: only maintain the best k models
    # use heapq (min heap) (heapify, heappush, heappop) 
    # if min heap item is tuple (i1, i2), will use the first value (i1) for cmp
    # https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/
    top_k_val = []
    heapq.heapify(top_k_val)    
    
    START_ITER = solver.START_ITER
    
    # used to enable tqdm progress display
    if not opt.hpc and opt.rank % opt.ngpus == 0:
        pbar = tqdm(total=opt.total_iterations-START_ITER) 
    
    for iter_id in range(START_ITER, opt.total_iterations):
        
        solver.train_iter(iter_id)
        
        if iter_id % 1000 == 999:
        # if iter_id % 10 == 9:
            if opt.rank % opt.ngpus == 0:
                print("====================================================")
                print("=> gpu {}: iteration finished: {}/{}".format(opt.gpu, iter_id, opt.total_iterations - START_ITER))
                
                # here val_loss is the abs_rel for gt_depth
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
                
                # abs_rel in DomainAda v.s. val_loss in PTNet
                print("=> current best models (-1*val_loss, iteration): \n{}".format(top_k_val))
        
        # used to enable tqdm progress display
        if not opt.hpc and opt.rank % opt.ngpus == 0:
            pbar.update(1)
    
    # close the program
    if not opt.hpc and opt.rank % opt.ngpus == 0:
        pbar.close()
    
    if opt.rank % opt.ngpus == 0:     
        solver.writer.close()
    