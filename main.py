from Solver import Solver
from options import SharinOptions
import os
import json
from tqdm import tqdm
import heapq


def check_opt(opt):
    
    T_OPT_CONSISTENCY = ["num_layers_T", "stereo_mode", "vbaseline", "predict_right_disp"]
    G_OPT_CONSISTENCY = ["num_layers_G", "netG_mode"]
    
    if opt.pretrained_model_T is not None:
        # e.g. "PTNet_Baseline/saved_models/tmp/PTNet_baseline-126999_bicubic.pth.tar"
        pretrained_T_opt = os.path.join("PTNet_Baseline/tensorboard_logs/vKitti2", opt.pretrained_model_T.split("/")[2], "PTNet_Baseline_bicubic/options/opt.json")
        assert os.path.isfile(pretrained_T_opt)
        
        with open(pretrained_T_opt, "r") as f:
            pre_T_opt = json.load(f) 
        
        for t_opt in T_OPT_CONSISTENCY:
            assert opt.__dict__[t_opt] == pre_T_opt[t_opt], "=> {} is inconsistent for pretrained_model_T: {}".format(t_opt, opt.pretrained_model_T)
        
        
    if opt.pretrained_model_G is not None:
        # e.g. "Gen_Baseline/saved_models/tmp/AE_Resnet_Baseline.pth.tar"
        pretrained_G_opt = os.path.join("Gen_Baseline/tensorboard_logs/vKitti2", opt.pretrained_model_G.split("/")[2], "AE_Baseline/Resnet_NEW/options/opt.json")
        assert os.path.isfile(pretrained_G_opt)
        

if __name__=='__main__':    
    options = SharinOptions()
    opt = options.parse()
    
    check_opt(opt)
    
    opt.gpu = 0
    opt.rank = 0
    opt.distributed = False
    
    solver = Solver(opt)
    
    solver.load_pretrained_models()
    if opt.resume:
        solver.load_prev_model()
        
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
        
            # print("====================================================")
            # print("=> gpu {}: iteration finished: {}/{}, sub_batch_size: {} ".format(opt.gpu, iter_id, opt.total_iterations - START_ITER, solver.batch_size))
            # val_loss = solver.val(iter_id).cpu().item()
            
            if opt.rank % opt.ngpus == 0:
                print("====================================================")
                print("=> gpu {}: iteration finished: {}/{}".format(opt.gpu, iter_id, opt.total_iterations - START_ITER))
                
                # here val_loss is the abs_rel for gt_depth
                val_loss = solver.val(iter_id)
                
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
                        
                print("=> current best models (-1*abs_rel, iteration): \n{}".format(top_k_val))
        
        # used to enable tqdm progress display
        if not opt.hpc and opt.rank % opt.ngpus == 0:
            pbar.update(1)
    
    # close the program
    if not opt.hpc and opt.rank % opt.ngpus == 0:
        pbar.close()
    
    if opt.rank % opt.ngpus == 0:     
        solver.writer.close()
    