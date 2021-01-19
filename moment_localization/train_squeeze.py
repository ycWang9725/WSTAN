from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import models.trans_loss as trans_loss
import math

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
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
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    train_dataset = getattr(datasets, dataset_name+'_squeeze')('train', unsup=config.DATASET.UNSUP)
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name+'_squeeze')('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name+'_squeeze')('val')
    test_dataset = getattr(datasets, dataset_name+'_squeeze')('test')

    # print(len(train_dataset))
    # print(len(test_dataset))
    # exit()

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint, strict=True)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999),
                           weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR,
                                                     patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)


    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_squeeze)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_squeeze)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_squeeze)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn_squeeze)
        else:
            raise NotImplementedError

        return dataloader


    def network(sample, warming=False):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        textual_sm_mask = sample['batch_txt_sm_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        neg_textual_input = sample['batch_neg_word_vectors'].cuda()
        neg_textual_mask = sample['batch_neg_txt_mask'].cuda()
        neg_textual_sm_mask = sample['batch_neg_txt_sm_mask'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']

        pos_scores, neg_scores, \
        attn_map, neg_attn_map = model(visual_input,
                                       textual_input,
                                       textual_mask, textual_sm_mask,
                                       neg_textual_input,
                                       neg_textual_mask, neg_textual_sm_mask)

        textual_sm_mask = textual_sm_mask.unsqueeze(-1)
        neg_textual_sm_mask = neg_textual_sm_mask.unsqueeze(-1)

        # print(torch.max(pos_scores), torch.max(neg_scores))
        # print(torch.max(attn_map), torch.max(neg_attn_map))
        # print(attn_map.shape)
        # exit()

        def compute_loss():

            scores = torch.sum(torch.sigmoid(pos_scores), dim=1) / torch.sum(textual_sm_mask, dim=1)\
                     - torch.sum(torch.sigmoid(neg_scores), dim=1) / torch.sum(neg_textual_sm_mask, dim=1)
            if torch.min(torch.sum(textual_sm_mask, dim=1)) == 0:
                print(textual_sm_mask.squeeze())
                print(textual_sm_mask.shape)
                # exit()
            # print(textual_sm_mask.shape)
            # print(torch.sum(textual_sm_mask, dim=1).squeeze(), torch.sum(neg_textual_sm_mask, dim=1).squeeze())

            joint_prob = torch.sigmoid(scores)
            # print(scores.shape)
            # pos_loss_fn = torch.nn.MSELoss(reduce=False)
            # pos_loss = pos_loss_fn(torch.sigmoid(attn_map), torch.ones_like(attn_map))
            # pos_loss = torch.sum(pos_loss * textual_sm_mask.squeeze().unsqueeze(1)) / torch.sum(textual_sm_mask)
            #
            # neg_loss_fn = torch.nn.MSELoss(reduce=False)
            # neg_loss = neg_loss_fn(torch.sigmoid(neg_attn_map), torch.zeros_like(neg_attn_map))
            # neg_loss = torch.sum(neg_loss * neg_textual_sm_mask.squeeze().unsqueeze(1)) / torch.sum(neg_textual_sm_mask)

            pos_prob = torch.sigmoid(torch.max(attn_map, dim=1)[0])
            pos_loss = torch.nn.functional.binary_cross_entropy(pos_prob, torch.ones_like(pos_prob), reduction='none')
            # print(attn_map.shape)
            # print(pos_prob[0])
            # print(pos_prob.shape)
            # print(textual_sm_mask.shape)
            pos_loss = torch.sum(pos_loss * textual_sm_mask.squeeze()) / torch.sum(textual_sm_mask)
            neg_prob = torch.sigmoid(torch.max(neg_attn_map, dim=1)[0])
            neg_loss = torch.nn.functional.binary_cross_entropy(neg_prob, torch.zeros_like(neg_prob), reduction='none')
            neg_loss = torch.sum(neg_loss * neg_textual_sm_mask.squeeze()) / torch.sum(neg_textual_sm_mask)

            loss_ = pos_loss + neg_loss
            # print(loss_, pos_loss, neg_loss)

            # return loss_, joint_prob

            map_mask = torch.zeros([1, 16, 16], dtype=int).cuda()
            for i in range(16):
                for j in range(i, 16):
                    if i <= j:
                        map_mask[:, i, j] = 1

            # joint_prob = torch.sigmoid(scores.squeeze()) * map_mask.squeeze()
            # joint_prob = joint_prob.flatten(-2)
            # flattened_map_gt = map_gt.flatten(-2)
            # target_prob = torch.zeros(flattened_map_gt.shape).cuda()
            # # print(torch.max(flattened_map_gt, dim=-1)[1])
            # for i in range(len(flattened_map_gt)):
            #     target_prob[i, torch.max(flattened_map_gt, dim=-1)[1][i]] = 1
            # # print(flattened_map_gt[1])
            # # print(target_prob[1])
            # # print(map_mask)
            # # print(joint_prob.shape, target_prob.shape)
            # loss = torch.nn.functional.binary_cross_entropy(joint_prob, target_prob, reduction='none') * map_mask.flatten(-2)
            # loss_value = torch.sum(loss) / torch.sum(map_mask)
            # # print(loss_value, joint_prob)
            # return loss_value, joint_prob

            if not torch.equal(scores, scores):
                print('nan in scores')
            loss_value, joint_prob = loss.bce_rescale_loss(scores, map_mask, map_gt, config.LOSS.PARAMS)

            loss_value = loss_value + loss_
            return loss_value, joint_prob

        loss_value, joint_prob = compute_loss()

        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)

        return loss_value, sorted_times


    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            sorted_indexs = np.dstack(
                np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

            sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

        return out_sorted_times


    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset) / config.TRAIN.BATCH_SIZE * config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])


    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)


    def on_update(state):  # Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n' + train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n' + val_table

            test_state = engine.test(network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message + table_message + '\n'
            logger.info(message)

            saved_model_filename = os.path.join(config.MODEL_DIR, '{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                dataset_name, model_name + '_' + config.DATASET.VIS_INPUT_TYPE,
                state['t'], test_state['Rank@N,mIoU@M'][0, 0], test_state['Rank@N,mIoU@M'][0, 1]))

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)

            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()


    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError


    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)
        #         for k, v in state.items():
        #             print('%s: %s' % (k, type(v)))

        #         exit()
        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)
        # print(sorted_segments)
        # exit()


    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        if dataset_name == 'DiDeMo':
            state['Rank@N,mIoU@M'], state['miou'] = eval.eval_didemo(state['sorted_segments_list'], annotations,
                                                                     verbose=False)
        else:
            state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations,
                                                                          verbose=False)
        if config.VERBOSE:
            state['progress_bar'].close()


    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)