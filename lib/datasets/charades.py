""" Dataset loader for the Charades-STA dataset """
import os
import csv

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

class Charades(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
    # print(word_embedding.num_embeddings)

    def __init__(self, split, unsup=False):
        super(Charades, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.unsup = unsup

        self.videos = []
        self.descriptions = []
        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])
        anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)), 'r')
        annotations = {}
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            self.videos.append(vid)
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            if s_time < e_time:
                # annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
                if vid not in annotations.keys():
                    annotations[vid] = []
                annotations[vid].append({'video': vid, 'times':[s_time, e_time], 'description': [sent], 'duration': self.durations[vid]})
        anno_file.close()

        # script
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
                vid, s_time, e_time = anno.split(" ")
                self.videos.append(vid)
                s_time = float(s_time)
                e_time = min(float(e_time), self.durations[vid])
                if s_time < e_time:
                    # annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
                    if vid not in annotations.keys():
                        annotations[vid] = []
                    annotations[vid].append({'video': vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
            anno_file.close()

            # print(len(self.videos))
            if unsup and self.split == 'train':
                # video_script = {}
                # with open(os.path.join(self.data_dir, "Charades_v1_{}.csv".format(self.split)),'r') as f:
                #     reader = csv.DictReader(f)
                #     for row in reader:
                #         vid = row['id']
                #         script = [sent.strip() for sent in row['script'].split('.')]
                #         video_script[vid] = []
                #         for item in script:
                #             if len(item.strip()) > 2:
                #                 video_script[vid].append(item.strip())

                self.annotations = []
                for vid, info in annotations.items():
                    # script = video_script[vid]
                    # if len(script) == 0:
                    #     print('no script for video {}'.format(vid))
                    #     continue
                    # print(script)
                    description = [item['description'] for item in info]
                    times = [item['times'] for item in info]
                    self.descriptions.append(description)
                    self.annotations.append({'video': vid, 'times': times, 'description': description, 'duration': self.durations[vid]})
                    # self.annotations.append({'video': vid, 'times': times, 'description': script, 'duration': self.durations[vid]})
                    # print()
            else:
                self.annotations = []
                for info in annotations.values():
                    for item in info:
                        self.annotations.append(item)
        # self.annotations = []
        # for info in annotations.values():
        #     for item in info:
        #         description = item['description']
        #         self.descriptions.append(description)
        #         self.annotations.append(item)
#             self.videos = list(set(self.videos))
        videos = list(set(self.videos))
        self.videos_features = {}
        for vid in videos:
            self.videos_features[vid] = self.get_video_features(vid)

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        times = self.annotations[index]['times']
        cur_description = self.annotations[index]['description']
        duration = self.durations[video_id]

        if type(cur_description) is list:
            if len(cur_description) > 1:
                description = random.sample(cur_description, len(cur_description) - 1)
            else:
                description = cur_description
            # description = cur_description
            # print(cur_description)
            # print(description)
            # exit()
            word_vectors = []
            txt_mask = []
            for sent in description:
                idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sent.split()], dtype=torch.long) 
                vectors = self.word_embedding(idxs)
                word_vectors.append(vectors)
                txt_mask.append(torch.ones(vectors.shape[0], 1))
            word_vectors = nn.utils.rnn.pad_sequence(word_vectors, batch_first=True)
            txt_mask = nn.utils.rnn.pad_sequence(txt_mask, batch_first=True)
            # print(cur_description)
            # print(txt_mask)
            # exit()
        else:
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in cur_description.split()], dtype=torch.long)
            word_vectors = self.word_embedding(word_idxs).unsqueeze(0)
            txt_mask = torch.ones(word_vectors.shape[0], 1).unsqueeze(0)

        # visual_input, visual_mask = self.get_video_features(video_id)
        visual_input, visual_mask = self.videos_features[video_id]

        # Time scaled to fixed size
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
                idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sent.split()], dtype=torch.long) 
                vectors = self.word_embedding(idxs)
                neg_word_vectors.append(vectors)
                neg_txt_mask.append(torch.ones(vectors.shape[0], 1))
            neg_word_vectors = nn.utils.rnn.pad_sequence(neg_word_vectors, batch_first=True)
            neg_txt_mask = nn.utils.rnn.pad_sequence(neg_txt_mask, batch_first=True)
            
            # neg_visual_input, neg_visual_mask = self.get_video_features(video_id)
            neg_visual_input, neg_visual_mask = self.videos_features[neg_video_id]
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
                #
                # print(gt_s_time, gt_e_time)
                # print(s_time, e_time)
                # print(overlap)
                # print(overlap.shape)
                # print(gt_s_idx, gt_e_idx)
                # exit()
        else:
            gt_s_time, gt_e_time = times
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
            s_times = torch.arange(0,num_clips).float()*duration/num_clips
            e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
            overlap = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                        e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips)

            overlaps = [overlap]
            gt_s_idx = np.argmax(overlap)//num_clips
            gt_e_idx = np.argmax(overlap)%num_clips

        # print(visual_input.shape)
        # exit()
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
