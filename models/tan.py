import torch
from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
from models.map_modules.transformer import TransformerDecoder, TransformerDecoderLayer
from models.map_modules.position_encoding import build_position_encoding

class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

        # refinements
        self.n_ref = config.TAN.N_REF
        if self.n_ref > 0:
            self.map_layer_1 = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
            self.pred_layer_1 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        if self.n_ref > 1:
            self.map_layer_2 = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
            self.pred_layer_2 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input):

        # print(visual_input.shape)
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        # print(vis_h.shape)
        # exit()
        map_h, map_mask = self.prop_layer(vis_h)
        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        mapped_h = self.map_layer(fused_h, map_mask)
        tmp_shape = mapped_h.shape
        # print(tmp_shape)
        mapped_h = torch.reshape(mapped_h, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))
        prediction = self.pred_layer(mapped_h)
        prediction = torch.reshape(prediction, (tmp_shape[0], tmp_shape[1],
                                                prediction.shape[-3],
                                                prediction.shape[-2],
                                                prediction.shape[-1]))
        # print(fused_h.shape)
        # print(prediction.shape)
        # print(map_mask.shape)
        prediction = prediction * map_mask.unsqueeze(1)

        # print(torch.max(prediction), torch.min(prediction))
        # merge sentences, get multi-modal paired info

        # merged_prediction, _ = torch.max(prediction, dim=1, keepdim=False)

        batch_size = textual_input.shape[0]
        sent_len = textual_input.shape[1]
        textual_mask = torch.reshape(textual_mask, (batch_size, sent_len, -1, 1))
        tmp_mask = (torch.sum(textual_mask, dim=(-2, -1), keepdim=True) > 0)
        # print(torch.sum(torch.unsqueeze(tmp_mask, -1), dim=1))
        merged_prediction = torch.sum(prediction, dim=1) / torch.sum(torch.unsqueeze(tmp_mask, -1), dim=1, dtype=torch.float)
        # print(merged_prediction.shape)
        # refinements
        if self.n_ref == 0:
            return merged_prediction, map_mask, [prediction]
        elif self.n_ref >= 1:
            # mapped_h_1 = self.map_layer_1(fused_h, map_mask)
            # tmp_shape = mapped_h_1.shape
            # mapped_h_1 = torch.reshape(mapped_h_1, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))

            prediction_1 = self.pred_layer_1(mapped_h)
            prediction_1 = torch.reshape(prediction_1, (tmp_shape[0], tmp_shape[1],
                                                        prediction_1.shape[-3],
                                                        prediction_1.shape[-2],
                                                        prediction_1.shape[-1]))
            prediction_1 = prediction_1 * map_mask.unsqueeze(1)
            if self.n_ref == 1:
                return merged_prediction, map_mask, [prediction, prediction_1]
            else:
                # mapped_h_2 = self.map_layer_2(fused_h, map_mask)
                # tmp_shape = mapped_h_2.shape
                # mapped_h_2 = torch.reshape(mapped_h_2, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))

                prediction_2 = self.pred_layer_2(mapped_h)
                prediction_2 = torch.reshape(prediction_2, (tmp_shape[0], tmp_shape[1],
                                                            prediction_2.shape[-3],
                                                            prediction_2.shape[-2],
                                                            prediction_2.shape[-1]))
                prediction_2 = prediction_2 * map_mask.unsqueeze(1)

                return merged_prediction, map_mask, [prediction, prediction_1, prediction_2]

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask


class trans_TAN(nn.Module):
    def __init__(self, nhead=8, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False):
        super(trans_TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        # self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        # self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)

        d_model = config.TAN.FUSION_MODULE.PARAMS.HIDDEN_SIZE
        self.text_map_layer = fusion_modules.TextMap(config.TAN.FUSION_MODULE.PARAMS)
        self.txt_posemb = nn.Embedding(config.TAN.MAP_MODULE.PARAMS.MAX_SEQ_LEN, d_model)
        self.position_embedding = build_position_encoding(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.map_layer = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)


        # refinements
        self.n_ref = config.TAN.N_REF
        if self.n_ref > 0:
            self.map_layer_1 = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
            self.pred_layer_1 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
        if self.n_ref > 1:
            self.map_layer_2 = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
            self.pred_layer_2 = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input):

        txt_h, txt_mask = self.text_map_layer(textual_input, textual_mask)
        # txt_pos = self.txt_posemb(txt_h)
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)
        # print(textual_input.shape)
        # print(map_h.shape)
        # print(map_mask.squeeze())
        # print(map_mask.shape)
        # exit()
        # fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        pos = self.position_embedding(map_mask.squeeze(1))
        # print(type(txt_h), txt_h.shape)
        # print(type(features), features.shape)
        # print(type(pos), pos.shape)
        # exit()
        mapped_h = []

        # print(pos)
        # print(txt_pos)
        # print(txt_h.shape)
        # exit()
        seq_length = txt_h.shape[-2]
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0)
        t_p = self.txt_posemb(position_ids)
        if not torch.equal(t_p, t_p):
            print('t_p')
            print(t_p)
        for t_h, t_m in zip(txt_h.transpose(0, 1), txt_mask.transpose(0, 1)):
            # print(t_m)
            mapped_h.append(self.map_layer(map_h, t_h, tgt_key_padding_mask=map_mask, pos=pos, memory_key_padding_mask=t_m, memory_pos=t_p))
        mapped_h = torch.stack(mapped_h).permute(1, 0, 4, 2, 3)
        if not torch.equal(mapped_h, mapped_h):
            print('mapped_h')
        tmp_shape = mapped_h.shape
        # print(tmp_shape)
        # exit()
        mapped_h = torch.reshape(mapped_h, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))

        prediction = self.pred_layer(mapped_h)

        prediction = torch.reshape(prediction, (tmp_shape[0], tmp_shape[1],
                                                prediction.shape[-3],
                                                prediction.shape[-2],
                                                prediction.shape[-1]))
        prediction = torch.squeeze(prediction * map_mask.unsqueeze(1), 2)

        # merge sentences, get multi-modal paired info
        # merged_prediction, _ = torch.max(prediction, dim=1, keepdim=False)

        batch_size = textual_input.shape[0]
        sent_len = textual_input.shape[1]
        textual_mask = torch.reshape(textual_mask, (batch_size, sent_len, -1, 1))
        tmp_mask = (torch.sum(textual_mask, dim=(-2, -1)) > 0)
        merged_prediction = torch.sum(prediction * tmp_mask.unsqueeze(-1).unsqueeze(-1), dim=1) / torch.sum(tmp_mask.unsqueeze(-1), dim=1, dtype=torch.float).unsqueeze(-1)
        if self.n_ref == 0:
            # print('2', prediction.shape)
            return merged_prediction, map_mask, [prediction]
        elif self.n_ref >= 1:
            # mapped_h_1 = self.map_layer_1(fused_h, map_mask)
            # tmp_shape = mapped_h_1.shape
            # mapped_h_1 = torch.reshape(mapped_h_1, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))

            prediction_1 = self.pred_layer_1(mapped_h)
            prediction_1 = torch.reshape(prediction_1, (tmp_shape[0], tmp_shape[1],
                                                        prediction_1.shape[-3],
                                                        prediction_1.shape[-2],
                                                        prediction_1.shape[-1]))
            prediction_1 = torch.squeeze(prediction_1 * map_mask.unsqueeze(1), 2)
            if self.n_ref == 1:
                return merged_prediction, map_mask, [prediction, prediction_1]
            else:
                # mapped_h_2 = self.map_layer_2(fused_h, map_mask)
                # tmp_shape = mapped_h_2.shape
                # mapped_h_2 = torch.reshape(mapped_h_2, (-1, tmp_shape[2], tmp_shape[3], tmp_shape[4]))

                prediction_2 = self.pred_layer_2(mapped_h)
                prediction_2 = torch.reshape(prediction_2, (tmp_shape[0], tmp_shape[1],
                                                            prediction_2.shape[-3],
                                                            prediction_2.shape[-2],
                                                            prediction_2.shape[-1]))
                prediction_2 = torch.squeeze(prediction_2 * map_mask.unsqueeze(1), 2)

                return merged_prediction, map_mask, [prediction, prediction_1, prediction_2]

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask
