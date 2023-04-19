from DVSO_Finetune.Solver_DVSO import Solver_DVSO
from options import SharinOptions
import os
import json
from tqdm import tqdm
import heapq

        
def train_dvso(opt):
    solver = Solver_DVSO(opt)
    solver.load_prev_model(opt.dvso_resume_exp, opt.dvso_resume_iter)
        
    # NOTE: Remain to do: only maintain the best k models
    # use heapq (min heap) (heapify, heappush, heappop) 
    # if min heap item is tuple (i1, i2), will use the first value (i1) for cmp
    # https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/
    top_k_val = []
    heapq.heapify(top_k_val)    
    
    # START_ITER = solver.START_ITER
    # NOTE: discard the training samples in the last batch
    # NOTE: use a bit more iters (to Nx1000)
    iters_per_epoch = len(solver.real_dataset) // solver.batch_size
    iters_per_epoch = (iters_per_epoch // 1000 + 1) * 1000

    total_id = 0
    clean_disp = False if opt.dvso_maintain_disp else True

    for epoch in range(opt.dvso_epochs):
        print("Epoch: {} => Prepare DVSO results based on current depth model...".format(epoch))
        print("Epoch: {} => Prepare the DVSO results of dvso_test_seqs...".format(epoch))
        if not opt.dvso_disp_exist:
            solver.prepare_dvso(epoch, phase="test")
            solver.run_dvso(epoch, clean_disp=clean_disp, phase="test", use_dvso_depth=False)
        if opt.dvso_test_only:
            solver.clean_epoch(epoch)
            solver.test_pose(epoch)
            break
        
        print("Epoch: {} => Prepare the DVSO results of dvso_train_seqs...".format(epoch))
        if not opt.dvso_disp_exist:
            solver.prepare_dvso(epoch, phase="train")
            solver.run_dvso(epoch, clean_disp=clean_disp, phase="train", use_dvso_depth=opt.use_dvso_depth)
        solver.update_epoch(epoch)
        
        # used to enable tqdm progress display
        if not opt.hpc and opt.rank % opt.ngpus == 0:
            pbar = tqdm(total=iters_per_epoch) 
        
        print("Epoch {}: Start DVSO consistency finetuning...".format(epoch))
        for iter_id in range(iters_per_epoch):
            solver.train_iter(total_id)
            total_id += 1
        
            if iter_id % 1000 == 999:
            # if iter_id % 10 == 9:             
                if opt.rank % opt.ngpus == 0:
                    print("====================================================")
                    print("=> epoch {} with gpu {}: iteration finished: {}/{}".format(epoch, opt.gpu, iter_id, iters_per_epoch))
                    
                    # here val_loss is the abs_rel for gt_depth
                    val_loss = solver.val(total_id)
                    
                    if len(top_k_val) < opt.top_k_val:
                        # -1* because heapq is min heap rather than max heap
                        heapq.heappush(top_k_val, (-1*val_loss, total_id, epoch))
                        solver.save_model(epoch, total_id)
                    else:
                        border_val_loss, border_iter, border_epoch = heapq.heappop(top_k_val)
                        border_val_loss *= -1
                        if val_loss < border_val_loss:
                            heapq.heappush(top_k_val, (-1*val_loss, total_id, epoch))
                            solver.rm_model(border_iter)
                            solver.save_model(epoch, total_id)
                        else:
                            heapq.heappush(top_k_val, (-1*border_val_loss, border_iter, border_epoch))
                            
                    print("=> current best models (-1*abs_rel, total_iteration, epoch): \n{}".format(top_k_val))
            
            # used to enable tqdm progress display
            if not opt.hpc and opt.rank % opt.ngpus == 0:
                pbar.update(1)

        solver.clean_epoch(epoch)

        # close the program
        if not opt.hpc and opt.rank % opt.ngpus == 0:
            pbar.close()
    
    if opt.rank % opt.ngpus == 0:     
        solver.writer.close()


def test_dvso(opt):
    solver = Solver_DVSO(opt)
    solver.load_test_model(opt.dvso_test_exp, opt.dvso_test_iter)
    if not opt.dvso_disp_exist:
        solver.prepare_dvso(0, phase="test", test_best=True)
    solver.run_dvso(0, clean_disp=True, phase="test", use_dvso_depth=False, test_best=True)


if __name__=='__main__':    
    options = SharinOptions()
    opt = options.parse()
    
    if opt.dvso_real_only and not opt.dvso_netT_only:
        raise ValueError("--dvso_real_only can only be used with --dvso_netT_only")
    
    opt.gpu = 0
    opt.rank = 0
    opt.distributed = False

    if opt.dvso_test_best:
        test_dvso(opt)
    else:
        train_dvso(opt)
    
    
    