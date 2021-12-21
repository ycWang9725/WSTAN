import torch
from torch import nn
import torch.nn.functional as F

class BaseFusion(nn.Module):

    def __init__(self, cfg):
        super(BaseFusion, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE
        self.txt_input_size = cfg.TXT_INPUT_SIZE
        self.txt_hidden_size = cfg.TXT_HIDDEN_SIZE
        self.textual_encoder = nn.LSTM(self.txt_input_size, self.txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else self.txt_hidden_size,
                                       num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
        self.tex_linear = nn.Linear(self.txt_hidden_size, hidden_size)
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)

    def forward(self, textual_input, textual_mask, map_h, map_mask):
        self.textual_encoder.flatten_parameters()
        batch_size = textual_input.shape[0]
        sent_len = textual_input.shape[1]
#         exit()
        textual_input = torch.reshape(textual_input, (batch_size * sent_len, -1, self.txt_input_size))
        textual_mask = torch.reshape(textual_mask, (batch_size * sent_len, -1, 1))
        txt_h = self.textual_encoder(textual_input)[0] * textual_mask
        txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)])
        txt_h = self.tex_linear(txt_h)[:,:,None,None]
        txt_h = torch.reshape(txt_h, (batch_size, sent_len, -1, self.txt_hidden_size))
        map_h = self.vis_conv(map_h)
        # print(txt_h.shape)
        # exit()
        fused_h = F.normalize(txt_h.squeeze(-2).unsqueeze(-1).unsqueeze(-1) * map_h.unsqueeze(1)) * map_mask.unsqueeze(1)
        # print(fused_h.shape)
        # print(map_h.shape)
        # exit()
        #
        # attn = nn.functional.softmax(map_h * txt_h, dim=1)
        # result = torch.sum(attn * fused_h, dim=1)
        #
        # result, _ = torch.max(fused_h, dim=1, keepdim=False)

        # textual_mask = torch.reshape(textual_mask, (batch_size, sent_len, -1, 1))
        # tmp_mask = (torch.sum(textual_mask, dim=(-2, -1), keepdim=True) > 0)
        # print(torch.sum(torch.unsqueeze(tmp_mask, -1), dim=1))
        # result = torch.sum(fused_h, dim=1) / torch.sum(torch.unsqueeze(tmp_mask, -1), dim=1, dtype=torch.float)
        
        result = fused_h
#         print(result)
        return result


class TextMap(nn.Module):
    def __init__(self, cfg):
        super(TextMap, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE
        self.txt_input_size = cfg.TXT_INPUT_SIZE
        self.txt_hidden_size = cfg.TXT_HIDDEN_SIZE
        # self.textual_encoder = nn.LSTM(self.txt_input_size,
        #                                self.txt_hidden_size // 2 if cfg.LSTM.BIDIRECTIONAL else self.txt_hidden_size,
        #                                num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL,
        #                                batch_first=True)
        self.tex_linear = nn.Linear(self.txt_input_size, self.txt_hidden_size)
        textual_encoder_layer = nn.TransformerEncoderLayer(d_model=self.txt_hidden_size, nhead=8)
        self.textual_encoder = nn.TransformerEncoder(textual_encoder_layer, num_layers=2)
        self.txt_posemb = nn.Embedding(cfg.MAX_SEQ_LEN, self.txt_hidden_size)


    def forward(self, textual_input, textual_mask):
        # self.textual_encoder.flatten_parameters()
        # print(textual_input.shape)
        # print(textual_mask.shape)
        batch_size = textual_input.shape[0]
        # sent_len = textual_input.shape[1]
        textual_input = torch.reshape(textual_input, (batch_size, -1, self.txt_input_size))
        textual_input = self.tex_linear(textual_input)
        textual_mask = torch.reshape(textual_mask, (batch_size, -1, 1))

        seq_length = textual_input.shape[-2]
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0)
        pos_emb = self.txt_posemb(position_ids)
        textual_input = textual_input + pos_emb
        # print(textual_input.shape)
        # print(pos_emb.shape)

        txt_h = self.textual_encoder(textual_input.permute(1, 0, 2),
                                     src_key_padding_mask=~(textual_mask.squeeze().bool()))
        # print(txt_h)
        # print(textual_mask.shape)
        # print(textual_mask)
        # txt_h = torch.stack([txt_h.permute(1, 0, 2)[i][torch.sum(mask).long() - 1]
        #                      for i, mask in enumerate(textual_mask.permute(1, 0, 2))])
        # txt_h = self.tex_linear(txt_h)[:, :, None, None]
        txt_h = torch.reshape(txt_h, (batch_size, -1, self.txt_hidden_size))
        textual_mask = torch.reshape(textual_mask, (batch_size, -1))
        # print(txt_h.shape)
        # print(textual_mask.shape)
        # print(textual_mask)
        # exit()
        return txt_h, textual_mask

