import torch
from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
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

