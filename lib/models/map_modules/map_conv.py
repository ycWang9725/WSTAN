import torch
from torch import nn
import torch.nn.functional as F
from models.map_modules import get_padded_mask_and_weight

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            if cfg.SQUEEZE:
                self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], [k, 1], s, p, d))
                self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], [1, k], s, p, d))
            else:
                self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, pred in enumerate(self.convs):
            tmp_shape = x.shape
            x = torch.reshape(x, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
#             print(tmp_shape, x.shape, masked_weight.shape)
            x = torch.reshape(x, (tmp_shape[0], tmp_shape[1], x.shape[-3], x.shape[-2], x.shape[-1]))
            masked_weight = masked_weight.unsqueeze(1)
            x = x * masked_weight
            # print(tmp_shape)
            # print(x.shape)
        return x

