""" Dataset loader for the ActivityNet Captions dataset """
import os
import json

import h5py
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from core.eval import iou
from core.config import config

class ActivityNet(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split, unsup=False):
        super(ActivityNet, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.unsup = unsup

        self.videos = []
        self.descriptions = []
        self.durations = {}

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(os.path.join(self.data_dir, '{}.json'.format(split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = {}
        for vid, video_anno in annotations.items():
            self.videos.append(vid)
            duration = video_anno['duration']
            self.durations[vid] = float(duration)
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    if vid not in anno_pairs.keys():
                        anno_pairs[vid] = []
                    anno_pairs[vid].append({'video': vid, 'duration': duration, 'times': [max(timestamp[0], 0), min(timestamp[1], duration)], 'description': sentence})

        if unsup and self.split == 'train':
            self.annotations = []
            for vid, info in anno_pairs.items():
                description = [item['description'] for item in info]
                times = [item['times'] for item in info]
                self.descriptions.append(description)
                self.annotations.append({'video': vid, 'times': times, 'description': description,
                                         'duration': self.durations[vid]})
        else:
            self.annotations = []
            for info in anno_pairs.values():
                for item in info:
                    # print(item)
                    # exit()
                    self.annotations.append(item)
        # self.annotations = []
        # for info in anno_pairs.values():
        #     for item in info:
        #         description = item['description']
        #         self.descriptions.append(description)
        #         self.annotations.append(item)
        self.videos = list(set(self.videos))
        videos = list(set(self.videos))
        self.videos_features = {}
        for vid in videos:
            self.videos_features[vid] = self.get_video_features(vid)

        # print(self.annotations[:10])
        # print(len(self.annotations))
        # print(len(anno_pairs))
        # print(len(annotations))
        # exit()

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        times = self.annotations[index]['times']
        cur_description = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        if type(cur_description) is list:
            # if len(cur_description) > 1:
            #     description = random.sample(cur_description, len(cur_description) - 1)
            # else:
            #     description = cur_description
            description = cur_description

            word_vectors = []
            txt_mask = []
            for sent in description:
                idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sent.split()], dtype=torch.long)
                vectors = self.word_embedding(idxs)
                word_vectors.append(vectors)
                txt_mask.append(torch.ones(vectors.shape[0], 1))
            word_vectors = nn.utils.rnn.pad_sequence(word_vectors, batch_first=True)
            txt_mask = nn.utils.rnn.pad_sequence(txt_mask, batch_first=True)
        else:
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in cur_description.split()], dtype=torch.long)
            word_vectors = self.word_embedding(word_idxs).unsqueeze(0)
            txt_mask = torch.ones(word_vectors.shape[0], 1).unsqueeze(0)

        # visual_input, visual_mask = self.get_video_features(video_id)
        visual_input, visual_mask = self.videos_features[video_id]

        # Time unscaled NEED FIXED WINDOW SIZE
        if config.DATASET.NUM_SAMPLE_CLIPS <= 0:
            # num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError
            # torch.arange(0,)

        # Time scaled to same size
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
        # visual_input = interpolate_to_fixed_length(visual_input)
        visual_input = average_to_fixed_length(visual_input)

        if self.unsup:
            neg_video_id = random.sample(self.videos, 1)[0]
            while neg_video_id == video_id:
                neg_video_id = random.sample(self.videos, 1)[0]

            sample_description = random.sample(self.descriptions, 1)[0]
            while sample_description == cur_description:
                sample_description = random.sample(self.descriptions, 1)[0]
            if len(sample_description) > 1:
                neg_description = random.sample(sample_description, len(sample_description) - 1)
            else:
                neg_description = sample_description

            neg_word_vectors = []
            neg_txt_mask = []
            for sent in neg_description:
                idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sent.split()],
                                    dtype=torch.long)
                vectors = self.word_embedding(idxs)
                neg_word_vectors.append(vectors)
                neg_txt_mask.append(torch.ones(vectors.shape[0], 1))
            neg_word_vectors = nn.utils.rnn.pad_sequence(neg_word_vectors, batch_first=True)
            neg_txt_mask = nn.utils.rnn.pad_sequence(neg_txt_mask, batch_first=True)

            # neg_visual_input, neg_visual_mask = self.get_video_features(neg_video_id)
            neg_visual_input, neg_visual_mask = self.videos_features[video_id]
            neg_visual_input = average_to_fixed_length(neg_visual_input)
        else:
            neg_visual_input = visual_input
            neg_word_vectors = word_vectors
            neg_txt_mask = txt_mask

        if type(times[0]) is list:
            overlaps = []
            gt_s_idx = []
            gt_e_idx = []
            for gt_s_time, gt_e_time in times:
                num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
                s_time = torch.arange(0, num_clips).float() * duration / num_clips
                e_time = torch.arange(1, num_clips + 1).float() * duration / num_clips
                overlap = iou(torch.stack([s_time[:, None].expand(-1, num_clips),
                                           e_time[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                              torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
                overlaps.append(overlap)
                gt_s_idx.append(np.argmax(overlaps) // num_clips)
                gt_e_idx.append(np.argmax(overlaps) % num_clips)
        else:
            gt_s_time, gt_e_time = times
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            s_times = torch.arange(0, num_clips).float() * duration / num_clips
            e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
            overlap = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                       e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                          torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)

            overlaps = [overlap]
            gt_s_idx = np.argmax(overlap) // num_clips
            gt_e_idx = np.argmax(overlap) % num_clips

        item = {
            'visual_input': visual_input,
            'neg_visual_input': neg_visual_input,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'txt_mask': txt_mask,
            'neg_word_vectors': neg_word_vectors,
            'neg_txt_mask': neg_txt_mask,
            'map_gt': torch.from_numpy(np.array(overlaps)),
            'reg_gt': torch.tensor([gt_s_idx, gt_e_idx]),
            'duration': duration
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid]['c3d_features'][:])
            # print(features.shape)
            # exit()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask