import os
import math
import argparse
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import _init_paths
from core.engine import Engine
import datasets
import models
from core.utils import AverageMeter
from core.config import config, update_config
from core.eval import eval_didemo, eval_predictions, display_results
import models.loss as loss

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # testing
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--split', default='val', required=True, choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose

def save_scores(scores, data, dataset_name, split):
    results = {}
    print(len(scores))
    for i, d in enumerate(data):
        vid = d['video']
        if vid not in results.keys():
            results[vid] = []
        results[vid].append((d['times'], scores[i]))
    save_dir = os.path.join(config.RESULT_DIR, dataset_name, '{}_{}_{}.pkl'.format(config.MODEL.NAME, config.DATASET.VIS_INPUT_TYPE, split))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, '{}_{}_{}.pkl'.format(config.MODEL.NAME, config.DATASET.VIS_INPUT_TYPE, split))
    pkl.dump(results,open(save_path,'wb'))

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = getattr(models, config.MODEL.NAME)()
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    # print(model.state_dict().keys())
    # print(model_checkpoint.keys())

    model.load_state_dict(model_checkpoint, strict=True)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    dataset_name = config.DATASET.NAME
    test_dataset = getattr(datasets, config.DATASET.NAME)(args.split)
    dataloader = DataLoader(test_dataset,
                            batch_size=config.TRAIN.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.WORKERS,
                            pin_memory=False,
                            collate_fn=datasets.collate_fn)

    def network(sample):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']

        merged_prediction, map_mask, pred_ks = model(textual_input, textual_mask, visual_input)

        loss_ks = []
        prob_ks = []
        for k in range(0, len(pred_ks)):
            # print(k)
            loss_k, joint_prob_k = loss.refinement_loss(pred_ks[k], map_mask, pred_ks[k - 1],
                                                        config.LOSS.PARAMS)
            loss_ks.append(loss_k)
            prob_ks.append(joint_prob_k)

        # print(len(loss_ks))
        loss_value = sum(loss_ks)
        sorted_times = get_proposal_results(prob_ks[-1].squeeze(1), duration)

        joint_prob = prob_ks[-1].squeeze().flatten(1).detach().cpu().numpy().tolist()
        # joint_prob = joint_prob.squeeze().flatten(1).detach().cpu().numpy().tolist()
        # return loss_value, sorted_times
        return loss_value, sorted_times, joint_prob

    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

            sorted_indexs[:,1] = sorted_indexs[:,1] + 1
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())
        # print(out_sorted_times)
        return out_sorted_times


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        state['scores_list'] = []
        state['output'] = []
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TRAIN.BATCH_SIZE))

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['scores_list'].extend([state['scores'][i] for i in batch_indexs])
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()
            print()

        annotations = test_dataset.annotations
        if dataset_name == 'DiDeMo':
            state['Rank@N,mIoU@M'], state['miou'] = eval_didemo(state['sorted_segments_list'], annotations,
                                                                     verbose=False)
        else:
            state['Rank@N,mIoU@M'], state['miou'] = eval_predictions(state['sorted_segments_list'], annotations,
                                                                          verbose=False)

        loss_message = '\ntest loss {:.4f}'.format(state['loss_meter'].avg)
        print(loss_message)
        state['loss_meter'].reset()
        test_table = display_results(state['Rank@N,mIoU@M'], state['miou'],
                                          'performance on testing set')
        table_message = '\n'+test_table
        print(table_message)

        # save_scores(state['sorted_segments_list'], annotations, config.DATASET.NAME, args.split)
        save_scores(state['scores_list'], annotations, config.DATASET.NAME, args.split)


    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network,dataloader, args.split)
