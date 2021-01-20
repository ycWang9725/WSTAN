""" Dataset loader for the Charades-STA dataset """
import os
import csv
import copy
import h5py
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

import nltk
from . import average_to_fixed_length
from core.eval import iou
from core.config import config

class Charades_squeeze(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
    # print(word_embedding.num_embeddings)

    def __init__(self, split, unsup=False):
        super(Charades_squeeze, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.unsup = unsup
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

        self.videos = []
        self.descriptions = []
        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])
        # anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)), 'r')
        # annotations = {}
        # for line in anno_file:
        #     anno, sent = line.split("##")
        #     sent = sent.split('.\n')[0]
        #     vid, s_time, e_time = anno.split(" ")
        #     self.videos.append(vid)
        #     s_time = float(s_time)
        #     e_time = min(float(e_time), self.durations[vid])
        #     if s_time < e_time:
        #         # annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
        #         if vid not in annotations.keys():
        #             annotations[vid] = []
        #         annotations[vid].append({'video': vid, 'times':[s_time, e_time], 'description': [sent], 'duration': self.durations[vid]})
        # anno_file.close()

        data_file = 'Charades_v1_{}.csv'
        with open(os.path.join(self.data_dir, data_file.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])
            anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)),'r')
            annotations = {}
            for line in anno_file:
                anno, sent = line.split("##")
                sent = sent.split('.\n')[0]
                sent_sm_mask = [1] * len(sent.split())
                # for i, w in enumerate(sent.split()):
                #     if w in self.stopwords:
                #         sent_sm_mask[i] = 0
                vid, s_time, e_time = anno.split(" ")
                self.videos.append(vid)
                s_time = float(s_time)
                e_time = min(float(e_time), self.durations[vid])
                if s_time < e_time:
                    # annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
                    if vid not in annotations.keys():
                        annotations[vid] = []
                    annotations[vid].append({'video': vid, 'times':[s_time, e_time], 'description': (sent, sent_sm_mask), 'duration': self.durations[vid]})
            anno_file.close()

        self.annotations = []
        for info in annotations.values():
            for item in info:
                description = item['description']
                self.descriptions.append(description)
                self.annotations.append(item)
        videos = list(set(self.videos))
        self.videos_features = {}
        # for vid in videos:
        #     self.videos_features[vid] = self.get_video_features(vid)

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        times = self.annotations[index]['times']
        description, sm_mask = self.annotations[index]['description']
        duration = self.durations[video_id]
        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs).unsqueeze(0)
        txt_sm_mask = torch.tensor(sm_mask).unsqueeze(-1)
        txt_mask = torch.ones(word_vectors.shape[1], 1).unsqueeze(0)
        # print(word_vectors.shape)
        # print(txt_mask.shape)
        # exit()

        visual_input, visual_mask = self.get_video_features(video_id)
        # visual_input, visual_mask = self.videos_features[video_id]

        # Time scaled to fixed size
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
        # visual_input = interpolate_to_fixed_length(visual_input)
        visual_input = average_to_fixed_length(visual_input)

        neg_description, neg_sm_mask = random.sample(self.descriptions, 1)[0]
        for i, w in enumerate(neg_description.split()):
            if w in description.split():
                neg_sm_mask[i] = 0
        while np.sum(neg_sm_mask) == 0:
            neg_description, neg_sm_mask = random.sample(self.descriptions, 1)[0]
            for i, w in enumerate(neg_description.split()):
                if w in description.split():
                    neg_sm_mask[i] = 0

        neg_word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in neg_description.split()],
                                 dtype=torch.long)
        neg_word_vectors = self.word_embedding(neg_word_idxs).unsqueeze(0)
        neg_txt_sm_mask = torch.tensor(neg_sm_mask).unsqueeze(-1)
        neg_txt_mask = torch.ones(neg_word_vectors.shape[1], 1).unsqueeze(0)


        gt_s_time, gt_e_time = times
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0,num_clips).float()*duration/num_clips
        e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
        overlap = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                    e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips)

        # overlaps = [overlap]
        gt_s_idx = np.argmax(overlap)//num_clips
        gt_e_idx = np.argmax(overlap)%num_clips

        # print(visual_input.shape)
        # print(overlap)
        # print(txt_sm_mask)
        # exit()
        item = {
            'visual_input': visual_input,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'txt_mask': txt_mask,
            'txt_sm_mask': txt_sm_mask,
            'neg_word_vectors': neg_word_vectors,
            'neg_txt_mask': neg_txt_mask,
            'neg_txt_sm_mask': neg_txt_sm_mask,
            'map_gt': torch.from_numpy(np.array(overlap)),
            'reg_gt': torch.tensor([gt_s_idx, gt_e_idx]),
            'duration': duration
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        hdf5_file = h5py.File(os.path.join(self.data_dir, '{}_features.hdf5'.format(self.vis_input_type)), 'r')
        features = torch.from_numpy(hdf5_file[vid][:]).float()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        # print(vid)
        # print(features.shape)
        # exit()
        return features, vis_mask

    def get_target_weights(self):
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        pos_count = [0 for _ in range(num_clips)]
        total_count = [0 for _ in range(num_clips)]
        pos_weight = torch.zeros(num_clips, num_clips)
        for anno in self.annotations:
            video_id = anno['video']
            gt_s_time, gt_e_time = anno['times']
            duration = self.durations[video_id]
            s_times = torch.arange(0, num_clips).float() * duration / num_clips
            e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
            overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                        e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
            overlaps[overlaps >= 0.5] = 1
            overlaps[overlaps < 0.5] = 0
            for i in range(num_clips):
                s_idxs = list(range(0, num_clips - i))
                e_idxs = [s_idx + i for s_idx in s_idxs]
                pos_count[i] += sum(overlaps[s_idxs, e_idxs])
                total_count[i] += len(s_idxs)

        for i in range(num_clips):
            s_idxs = list(range(0, num_clips - i))
            e_idxs = [s_idx + i for s_idx in s_idxs]
            # anchor weights
            # pos_weight[s_idxs,e_idxs] = pos_count[i]/total_count[i]
            # global weights
            pos_weight[s_idxs, e_idxs] = sum(pos_count) / sum(total_count)


        return pos_weight
