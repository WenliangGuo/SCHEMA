import numpy as np
import torch
import torch.distributed as dist


def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy(output, target, topk=(1,), max_traj_len=0):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # Accuracy
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [5, 1620]

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        correct_1 = correct[:1]  # .view(-1, max_traj_len) # (bz, 1)

        # Success Rate
        trajectory_success = torch.all(correct_1.view(-1, max_traj_len), dim=1)
        trajectory_success_rate = trajectory_success.sum() * 100.0 / trajectory_success.shape[0]

        # MIoU
        _, pred_token = output.topk(1, 1, True, True)
        pred_inst = pred_token.view(-1, max_traj_len)
        pred_inst_set = set()
        target_inst = target.view(-1, max_traj_len)
        target_inst_set = set()
        MIoU_list = []
        for i in range(pred_inst.shape[0]):
            pred_inst_set.update(pred_inst[i].tolist())
            target_inst_set.update(target_inst[i].tolist())
            inter = pred_inst_set.intersection(target_inst_set)
            union = pred_inst_set.union(target_inst_set)
            MIoU = 100.0 * len(inter) / len(union)
            MIoU_list.append(MIoU)
        MIoU = sum(MIoU_list) / len(MIoU_list)

        return res, trajectory_success_rate, MIoU


def step_accuracy(output, target):
    pred = torch.argmax(output, dim=-1)
    with torch.no_grad():
        # Accuracy
        batch_size = target.size(0)
        step_acc = torch.sum(torch.eq(pred, target), dim=0) * (100/ batch_size)
        step_acc = step_acc.tolist()
    return step_acc


def step_success(output, target):
    pred = torch.argmax(output, dim=-1)
    batch_size, n_steps = target.shape
    success_list = [0] * (n_steps+1)
    with torch.no_grad():
        # Accuracy
        step_success = torch.sum(torch.eq(pred, target), dim=1)
    for i in range(n_steps+1):
        success_list[i] = sum(step_success==i)

    return success_list


def success_rate(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: The predicted intermediate action sequence, numpy [batch, seq];
    gt  : The ground-truth action label sequence    , numpy [batch, seq];

    Metric Procedure:
    "All" prediction steps has to match with gt steps
    """
    rst = np.all(np.equal(pred, gt), axis=(1))

    if aggregate:
        return np.mean(rst) * 100
    else:
        return rst


def mean_category_acc(pred, gt):
    """required format
    Action space is a single integer
    pred: List [batch * seq]
    gt  : List [batch * seq]
    """
    # rst = precision_score(gt, pred, average="macro", zero_division=0)
    rst = (gt == pred).mean() * 100
    return rst


def acc_iou(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: Numpy [batch, seq]
    gt  : Numpy [batch, seq]
    """
    epsn = 1e-6

    if aggregate:
        intersection = (pred & gt).sum((0, 1))
        union = (pred | gt).sum((0, 1))
    else:
        intersection = (pred & gt).sum((1))
        union = (pred | gt).sum((1))
        
    return (intersection + epsn) / (union + epsn) * 100


def reduce_metric(met_single):
    """
    Reduce the metrics from all processes so that process with rank
    0 has the averaged results. Returns a metric after reduction.
    """

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        return met_single

    if world_size < 2:
        return met_single
    
    with torch.no_grad():
        dist.reduce(met_single, dst=0)
        reduced_met = met_single
        if dist.get_rank() == 0:
            reduced_met = reduced_met / world_size

    return reduced_met