# -*- coding:utf-8 -*-
import os
import pickle
import json

import numpy as np
from matplotlib import pyplot as plt

# import cv2
import os
from datetime import datetime


def delAll(path):
    if os.path.isdir(path):
        files = os.listdir(path)  # ['a.doc', 'b.xls', 'c.ppt']
        # 遍历并删除文件
        for file in files:
            p = os.path.join(path, file)
            if os.path.isdir(p):
                # 递归
                delAll(p)
            else:
                os.remove(p)
        # 删除文件夹
        os.rmdir(path)
    else:
        os.remove(path)


def video_to_frames(dir, video_name):

    video_path = os.path.join(dir, '{}.mp4'.format(video_name))
    print(video_path)
    videoCapture = cv2.VideoCapture(video_path)
    rval, frame = videoCapture.read()
    if rval:
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("fps=", int(fps), "frames=", int(frames))

        save_path = os.path.join(dir, '{}_frames'.format(video_name))
        if os.path.exists(save_path):
            delAll(save_path)
        os.mkdir(save_path)
        for i in range(int(frames) - 1):
            ret, frame = videoCapture.read()
            cv2.imwrite(os.path.join(save_path, "frames%d.jpg" % (i)), frame)
    else:
        print(rval)

    return


if __name__ == '__main__':

    def extract_frames():
        t1 = datetime.now()
        # dir = '/data1/wangyc/2D-TAN/data/Charades-STA/videos'
        dir = 'data/didemo/videos/'
        videos = ['dog_video']
        # videos = ['Y1HGC', 'YZ8HK', '2RFLZ', 'N3U9S', 'V1WN7', 'J9T5D'， '1ZWPP']
        for video_name in videos:
            video_to_frames(dir, video_name)
        t2 = datetime.now()
        print("Time cost = ", (t2 - t1))
        print("SUCCEED !!!")

    def show_pkl(path, name):

        with open(os.path.join(path, '{}_result.pkl'.format(name)), 'rb') as f:
            data = pickle.load(f)

        # print(data['26292851@N04_3863871139_37e6344292.m4v'])
        print(len(data))

        with open(os.path.join(path, '{}_result.json'.format(name)), 'w') as f:
            json.dump(data, f)

    def merge_pred():
        def readpred(pred_file, name):
            pred = {}
            with open(pred_file, 'rb') as f:
                file_preds = json.load(f)
                for vid, des_list in file_preds.items():
                    pred[vid] = []
                    for gt, pred_rank in des_list:
                        pred[vid].append({
                            'gt': gt,
                            # 'pred': pred_rank[0]
                            'pred': pred_rank
                        })
            return pred


        pred_dir = 'results/Charades/TAN_vgg_rgb_test.pkl'
        base_file = os.path.join(pred_dir, 'base_result.json')
        self_file = os.path.join(pred_dir, 'self_result.json')
        full_file = os.path.join(pred_dir, 'full_result.json')

        base_pred = readpred(base_file, 'base')
        self_pred = readpred(self_file, 'self')
        full_pred = readpred(full_file, 'full')

        pred = {}
        for vid in base_pred.keys():
            pred[vid] = [{'gt': base['gt'], 'base': base['pred'], 'self':self['pred'], 'full': full['pred']} for base, self, full in zip(base_pred[vid], self_pred[vid] , full_pred[vid])]

        with open(os.path.join(pred_dir, 'merged_pred.json'), 'w') as fw:
            json.dump(pred, fw, indent=4)

    def visualize_heatmap():
        def draw(data, size, title, save_path=None):
            xLabel = range(size)
            yLabel = range(size)
            # fron_size = 16
            fig = plt.figure()
            ax = fig.add_subplot(131)
            plt.xticks()
            plt.yticks()
            im = ax.imshow(data[0], cmap=plt.cm.jet)
            # cb = plt.colorbar(im)
            # cb.set_label('colorbar', fontsize=fron_size)

            ax = fig.add_subplot(132)
            plt.xticks()
            plt.yticks()
            im = ax.imshow(data[1], cmap=plt.cm.jet)
            # cb = plt.colorbar(im)
            # cb.set_label('colorbar', fontsize=fron_size)

            ax = fig.add_subplot(133)
            plt.xticks()
            plt.yticks()
            im = ax.imshow(data[2], cmap=plt.cm.jet)
            # cb = plt.colorbar(im)
            # cb.set_label('colorbar', fontsize=fron_size)
            plt.tight_layout()
            if save_path is not None:
                # plt.savefig(save_path+'.eps', dpi=600, format='eps')
                # plt.savefig(save_path+'.png')
                plt.savefig(save_path+'.pdf', format='pdf')
                print(save_path)
            else:
                plt.show()
            plt.close()

        def masked_normalize(x):
            size = x.shape[0]
            mask = []
            for i in range(16):
                temp = []
                for j in range(16):
                    temp.append(1 if j>=i else 0)
                mask.append(temp)
            mask = np.array(mask)
            num = np.sum(mask)
            mu = np.sum(x) / num
            std = np.sqrt(np.sum(np.square(x-mu) * mask)/num)
            # print(mu)
            # print(std)
            x = (x - mu) * mask / std
            x[mask==0] = np.min(x)
            # return x
            Max = np.max(x)
            Min = np.min(x)
            return (x - Min) / (Max - Min)

        def sigmoid_r(x):
            x = np.log(x/(1-x))
            return x
            Max = np.max(x)
            Min = np.min(x)
            return (x - Min) / (Max - Min)

        # import random
        # data = []
        # for i in range(16):
        #     temp = []
        #     for j in range(16):
        #         k = random.randint(0, 100)
        #         temp.append(k)
        #     data.append(temp)
        # data = np.array(data)
        # # print(data.shape)
        # path = 'E:\yuechen\Projects\\2D-TAN\\results\Charades\\visualized\\visualized_pdf'
        # draw((data, data, data), 16, 'test', save_path=os.path.join(path, 'test'))
        # exit()
        path = 'E:\yuechen\Projects\\2D-TAN\\results\Charades\\TAN_vgg_rgb_test.pkl\\'
        with open(os.path.join(path, 'merged_pred.json')) as f:
            data = json.load(f)
        size = 16
        for vid, info_list in data.items():
            for i, info in enumerate(info_list):
                gt = info['gt']
                base_pred = np.array(info['base']).reshape([size, size])
                self_pred = np.array(info['self']).reshape([size, size])
                full_pred = np.array(info['full']).reshape([size, size])
                # print(base_pred)
                base_pred = sigmoid_r(base_pred)
                self_pred = sigmoid_r(self_pred)
                full_pred = sigmoid_r(full_pred)
                # print(base_pred)
                # exit()
                title = 'vid:%s gt:%s' % (vid, gt)
                draw((base_pred, self_pred, full_pred), size, title, save_path=os.path.join('E:\yuechen\Projects\\2D-TAN\\results\Charades\\visualized\\newcolor_pdf', 'heatmap_%s_%s' % (vid, i)))

    def find_example():
        def sigmoid_r(x):
            x = np.log(x / (1 - x))
            return x
            Max = np.max(x)
            Min = np.min(x)
            return (x - Min) / (Max - Min)

        path = 'E:\yuechen\Projects\\2D-TAN\\results\Charades\\TAN_vgg_rgb_test.pkl\\'
        with open(os.path.join(path, 'merged_pred.json')) as f:
            data = json.load(f)
        size = 16
        for vid, info_list in data.items():
            for i, info in enumerate(info_list):
                gt = info['gt']
                base_pred = np.array(info['base']).reshape([size, size])
                # self_pred = np.array(info['self']).reshape([size, size])
                full_pred = np.array(info['full']).reshape([size, size])
                # print(base_pred)
                # base_pred = sigmoid_r(base_pred)
                # self_pred = sigmoid_r(self_pred)
                # full_pred = sigmoid_r(full_pred)
                m, n = divmod(np.argmax(base_pred), size)
                x, y = divmod(np.argmax(full_pred), size)
                # exit()
                if vid=='2RFLZ':
                    print(vid, i, (m, n), (x, y), gt)

    # extract_frames()

    # path = '/data1/wangyc/2D-TAN/results/ActivityNet/TAN_c3d_test.pkl'
    # path = '/data1/wangyc/2D-TAN/results/Charades/TAN_vgg_rgb_test.pkl'
    # name = 'self'
    # show_pkl(path, name)
    # merge_pred()
    visualize_heatmap()
    # find_example()
