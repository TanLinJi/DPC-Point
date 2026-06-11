import os
import sys
import wandb

import torch
import torch.nn.functional as F
import operator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, shot_capacity, include_prob_map=False):
    """Build cache with new features and loss, maintaining the maximum shot capacity."""
    '''
    NOTE 1. for positive cache, `include_prob_map=False`
                feature_loss = [pc_feats, loss]

         2. for negative cache, `include_prob_map=True`
                feature_loss = [pc_feats, loss, prob_map]
    '''
    # NOTE `cache` is empty initially
    if include_prob_map:
        print('*'*10, 'Building neg. cache ...', '*'*10, '\n')
    else:
        print('*'*10, 'Building pos. cache ...', '*'*10, '\n')
    cache = {}
    
    for pc, _, _, rgb in test_loader:
        # pc: (1, n, 3)     rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()
        
        # `pred` indicates the class index
        pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        
        item = [pc_feats, loss] if not include_prob_map else [pc_feats, loss, prob_map]
        
        if pred in cache:   
            if len(cache[pred]) < shot_capacity:
                # cache[pred] -> a list of: 
                #   [[feats, loss], ..., [feats, loss]] or [[feats, loss, prob_map], ..., [feats, loss, prob_map]]
                cache[pred].append(item)
        else:
            cache[pred] = [item]
            
        cache_num = sum([len(cache[key]) for key in cache])
        # clip_logits: (1, num_classes)
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes
        
        # NOTE check if cache is full
        if cache_num == full_num:
            if include_prob_map:
                print('*'*10, 'Building neg. cache is Done!', '*'*10, '\n')
            else:
                print('*'*10, 'Building pos. cache is Done!', '*'*10, '\n')
            break
        
    return cache
            

@torch.no_grad()
def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    '''
    NOTE 1. for positive cache, `include_prob_map=False`
                feature_loss = [pc_feats, loss]

         2. for negative cache, `include_prob_map=True`
                feature_loss = [pc_feats, loss, prob_map]
    '''
    item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
    # NOTE `cache` is empty initially
    #   `pred` indicates the class index
    if pred in cache:   
        if len(cache[pred]) < shot_capacity:
            # cache[pred] -> a list of: 
            #   [[feats, loss], ..., [feats, loss]] or [[feats, loss, prob_map], ..., [feats, loss, prob_map]]
            cache[pred].append(item)
        elif features_loss[1] < cache[pred][-1][1]:
            # NOTE *** if the cache is full, and current `feats` has smaller entropy, \
            #   then replace the last item of cache with current `feats`
            cache[pred][-1] = item
        cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
    else:
        cache[pred] = [item]


@torch.no_grad()
def compute_cache_logits(pc_feats, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    cache_keys = []
    cache_values = []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            # item[0] -> `pc_feats` of shape (1, emb_dim)
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                # NOTE *** for negative cache, item[2] -> `prob_map` is a class probability distribution
                cache_values.append(item[2])
            else:
                # NOTE class_index is an integer
                cache_values.append(class_index)

    # cache_keys: (emb_dim, n_cls * k_shot)
    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

    if neg_mask_thresholds:
        # (n_cls * k_shot, emb_dim)
        cache_values = torch.cat(cache_values, dim=0)
        cache_values = ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).half().cuda()
    else:
        # NOTE 
        # 1. clip_weights: (emb_dim, n_cls)
        # 2. cache_values: (n_cls * k_shot, n_cls)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).half().cuda()

    # affinity: (1, n_cls * k_shot)
    affinity = pc_feats @ cache_keys
    # cache_logits: (1, n_cls)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    ''' NOTE Build cache in advance '''
    pos_cache = build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, pos_cfg['shot_capacity'])
    print('len(pos_cache):', len(pos_cache))
    neg_cache = {}
    
    accuracies = []
    
    #Unpack all hyperparameters
    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

    #Test-time adaptation
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        # pc: (1, n, 3)     rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()

        # pc_feats: (1, emb_dim)
        # clip_logits: (1, n_cls)
        # loss: a scalar
        # prob_map: (1, n_cls)
        # pred: a scalar, class index
        pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        # NOTE *** why compute `get_entropy()` ??? -> scale `loss` with `log_2(n_cls)`
        #   what's the difference between `get_entropy()` and `avg_entropy()` ???
        # target: (1, )     prop_entropy: (1, )
        target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

        if pos_enabled:
            update_cache(pos_cache, pred, [pc_feats, loss], pos_params['shot_capacity'])

        if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
            update_cache(neg_cache, pred, [pc_feats, loss, prob_map], neg_params['shot_capacity'], True)

        final_logits = clip_logits.clone()
        if pos_enabled and pos_cache:
            final_logits += compute_cache_logits(pc_feats, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(pc_feats, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
            
        acc = cls_acc(final_logits, target)  
        accuracies.append(acc)
        wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
    print("---- ***Final*** TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
    return sum(accuracies)/len(accuracies)


def main():
    args = get_arguments()
    
    print('#'*20)
    print('\targs.seed:', args.seed)
    print('#'*20)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # NOTE config
    config_path = args.config

    clip_model, lm3d_model = load_models(args)

    # NOTE *** need to be implemented
    preprocess = None

    # Run TDA on each dataset
    dataset_name = args.dataset
    print(f"Processing {dataset_name} dataset.")
    
    cfg = get_config_file(args, config_path, dataset_name)
    print("\nRunning dataset configurations:")
    print(cfg, "\n")
    
    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
    
    print(f'>>> classnames:', classnames)
    
    # `clip_weights` are text features of shape (emb_dim, n_cls)
    clip_weights = clip_classifier(args, classnames, template, clip_model)

    if args.wandb:
        if args.lm3d == 'openshape':
            prefix = f"[ablate-seed{args.seed}]/{args.cache_type}_cache/{args.lm3d}-{args.oshape_version}" 
        elif args.lm3d == 'ulip':
            prefix = f"[ablate-seed{args.seed}]/{args.cache_type}_cache/{args.ulip_version}"
        else:
            prefix = f"[ablate-seed{args.seed}]/{args.cache_type}_cache/{args.lm3d}"
        
        if '_c' in dataset_name and 'sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}/{args.cor_type}"
        elif '_c' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
        elif 'scanobjnn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}"
        elif 'sim2real_sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sim2real_type}-{args.npoints}"
        elif 'pointda' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        else:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        
        run = wandb.init(project="Point-TDA", config=cfg, name=run_name)

    acc = run_test_tda(args, cfg['positive'], cfg['negative'], test_loader, lm3d_model, clip_weights)

    if args.wandb:
        wandb.log({f"{dataset_name}": acc})
        run.finish()


if __name__ == "__main__":
    main()