import torch
import torch.nn.functional as F


def bce_rescale_loss(scores, masks, targets, cfg):
    # print(targets.shape)
    # print(scores.shape)
    # print(masks.shape)
    # exit()
    # targets = torch.tensor(targets == 1)
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores.squeeze()) * masks.squeeze()
    target_prob = (targets.squeeze() - min_iou) * (1 - bias) / (max_iou - min_iou)
    # print(target_prob)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    # print(joint_prob.shape, target_prob.shape)
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    loss_value = torch.sum(loss) / torch.sum(masks)
    # print(loss_value, joint_prob)
    return loss_value, joint_prob


def self_restrict_loss(scores, masks, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    # print(scores.shape)
    # print(masks.shape)
    # masks = masks.unsqueeze(1)
    joint_prob = torch.sigmoid(scores) * masks

    tmp_shape = scores.shape
    # print(joint_prob.flatten(-2).shape)
    weight_1, targets_tmp = torch.max(joint_prob.flatten(-2), dim=-1)
    weight_1_detached = weight_1.detach()
    targets_tmp_detached = targets_tmp.detach()
    # print(weight_1)
    # print(targets_tmp)
    targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
    targets.scatter_(2, targets_tmp_detached.unsqueeze(-1), 1)
    targets = torch.reshape(targets, tmp_shape)

    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    # print("joint_prob", joint_prob.shape)
    # print("target_prob", target_prob.shape)
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    # print(joint_prob.shape)
    # print(target_prob.shape)
    # print(loss.shape)
    # print(masks.shape)
    loss_value = torch.sum(loss * weight_1_detached.unsqueeze(-1).unsqueeze(-1)) / torch.sum(masks)
    return loss_value, joint_prob


def refinement_loss(scores_1, masks, scores, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    # masks = masks.unsqueeze(1)
    joint_prob = torch.sigmoid(scores_1) * masks

    tmp_shape = scores.shape
    targets = torch.sigmoid(scores) * masks
    targets = targets.flatten(-2)
    # with torch.no_grad():
    weight_1, targets_tmp = torch.max(targets, dim=-1)
    weight_1_detached = weight_1.detach()
    targets_tmp_detached = targets_tmp.detach()

    targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
    targets.scatter_(2, targets_tmp_detached.unsqueeze(-1), 1)
    targets = torch.reshape(targets, scores_1.shape)

    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    # print("joint_prob", joint_prob.shape)
    # print("target_prob", target_prob.shape)
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks.unsqueeze(1)
    # print(loss.shape, scores.shape)
    # print("loss", loss)
    # print(scores)
    # print(loss.shape, weight.shape)
    # print(scores_1.shape)
    # print(weight_1_detached.shape)
    # print(loss.shape)
    # loss_value = torch.sum(loss) / torch.sum(masks)
    loss_value = torch.sum(loss * weight_1_detached.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / torch.sum(masks)
    # print("loss_value", loss_value)
    # exit()
    return loss_value, joint_prob


def soft_refinement_loss(scores_1, masks, scores, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores_1) * masks.unsqueeze(1)

    tmp_shape = scores.shape
    targets = torch.sigmoid(scores) * masks.unsqueeze(1)
    targets = torch.reshape(targets, (tmp_shape[0], tmp_shape[1], -1))
    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 0.5] = 1
    target_prob[target_prob <= 0.5] = 0

    loss = F.smooth_l1_loss(joint_prob, target_prob, size_average=False)

    loss_value = torch.sum(loss) / torch.sum(masks)
    # print("loss_value", loss_value)
    # exit()
    return loss_value, joint_prob


def topk_refinement_loss(scores_1, masks, scores, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores_1) * masks.unsqueeze(1)

    tmp_shape = scores.shape
    scores = torch.sigmoid(scores) * masks.unsqueeze(1)
    scores = torch.reshape(scores, (tmp_shape[0], tmp_shape[1], -1))
    weight_1, targets_tmp = torch.topk(scores, k=5, dim=-1)

    targets = torch.zeros(tmp_shape[0], tmp_shape[1], tmp_shape[-2] * tmp_shape[-1]).cuda()
    targets.scatter_(2, targets_tmp, 1)
    targets = torch.reshape(targets, scores_1.shape)

    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks.unsqueeze(1)
    loss_value = torch.sum(loss * weight_1.mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / torch.sum(masks)
    return loss_value, joint_prob


def triplet_loss(pos_scores, pos_masks, neg_v_scores, neg_v_masks, neg_t_scores, neg_t_masks, cfg):
    delta = cfg.DELTA
    n_neg = cfg.N_NEG

    # print(pos_scores)
    def cal_score(scores, masks):
        score = torch.sigmoid(scores) * masks
        score, _ = torch.max(score.flatten(1), dim=-1)
        # score = torch.sum(torch.sigmoid(scores) * masks, dim=(1, 2, 3)) / torch.sum(masks, dim=(1, 2, 3))
#         score = torch.sum(scores * masks, dim=(1, 2, 3)) / torch.sum(masks, dim=(1, 2, 3))
#         print(torch.sigmoid(scores) * masks)
#         print(torch.sum(masks, dim=(1, 2, 3)))
        # exit()
        return score
    
    pos_score = cal_score(pos_scores, pos_masks)
    neg_v_score = cal_score(neg_v_scores, neg_v_masks)
    neg_t_score = cal_score(neg_t_scores, neg_t_masks)
    # print(torch.stack((pos_score, neg_v_score, neg_t_score), dim=-1))
    batch_size = pos_score.shape[0]
    # if not cfg.CROSS_ENTROPY:
    #     tmp_0 = torch.tensor([[0] * batch_size], dtype=torch.float).cuda()
    #     loss = torch.sum(torch.max(tmp_0, delta - pos_score + neg_t_score) + torch.max(tmp_0, delta - pos_score + neg_v_score))
    # else:
    loss = 0
    target_prob = torch.tensor([1] * batch_size, dtype=torch.float).cuda()
    # print(pos_score)
    # print(target_prob)
    loss += torch.sum(F.binary_cross_entropy(pos_score, target_prob, reduction='none'))
    target_prob = torch.tensor([0] * batch_size, dtype=torch.float).cuda()
    loss += torch.sum(F.binary_cross_entropy(neg_v_score, target_prob, reduction='none'))
    if n_neg == 2:
        loss += torch.sum(F.binary_cross_entropy(neg_t_score, target_prob, reduction='none'))
#     print(loss)
#     exit()
    return loss
