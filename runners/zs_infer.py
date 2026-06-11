import os
import sys
import wandb

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


def infer(args, lm3d_model, test_loader, clip_weights):
    # assert args.cache_type == 'local', f'Local cache is expected, but got {args.cache_type}!'
    
    print('>>> In function `run`: zero-shot inference')
    
    accuracies = []
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        # pc: (1, n, 3)     rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()
        target = target.cuda()
        
        # pc_feats: (1, emb_dim)
        # patch_centers: (5, emb_dim)
        # clip_logits: (1, n_cls)
        # loss: a scalar
        # prob_map: (1, n_cls)
        # pred: a scalar, class index
        if args.cache_type == 'local':
            patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        elif args.cache_type == 'global':
            pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        elif args.cache_type == 'hierarchical':
            pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        else:
            raise ValueError(f'The choice from [local, global, hierarchical] is expected, but got {args.cache_type}!')
            
        acc = cls_acc(clip_logits, target)
        accuracies.append(acc)
        
        wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)
        if i % args.print_freq == 0:
            print("---- Zero-shot test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        
    print("---- ***Final*** Zero-shot test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies))) 
    return sum(accuracies)/len(accuracies)


def main(args):
    print('>>> In function `main`')
    
    clip_model, lm3d_model = load_models(args)
    
    preprocess = None
    
    # Run TDA on each dataset
    dataset_name = args.dataset
    print('>>> In loop `for`')
    
    print(f"Processing {dataset_name} dataset.")
    
    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
    
    print(f'>>> {[dataset_name]} classnames: {classnames} \n')
    
    # `clip_weights` are text features of shape (emb_dim, n_cls)
    clip_weights = clip_classifier(args, classnames, template, clip_model)
    
    if args.wandb:
        if args.lm3d == 'openshape':
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}-{args.oshape_version}"
        elif args.lm3d == 'ulip':
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.ulip_version}"
        else:
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}"
        
        if '_c' in dataset_name and 'sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}/{args.cor_type}"
        elif '_c' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
        elif 'scanobjnn' in dataset_name or 'scanobjectnn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}"
        elif 'sim2real_sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sim2real_type}-{args.npoints}"
        elif 'pointda' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        else:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        
        run = wandb.init(project="Point-TDA", name=run_name)

    zs_acc = infer(args, lm3d_model, test_loader, clip_weights)
    
    if args.wandb:
        wandb.log({f"{dataset_name}": zs_acc})
        run.finish()
        
        
if __name__ == '__main__':
    args = get_arguments()
    # Set random seed
    set_random_seed(args.seed)
    
    main(args)
