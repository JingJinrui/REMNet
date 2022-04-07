import torch as t
from skimage.metrics import structural_similarity


def crosstab_evaluate(output, ground_truth, dBZ_downvalue, dBZ_upvalue, dataset='HKO_7'):
    if dataset == 'HKO_7':
        dBZ_output = 70.0 * output - 10.0
        dBZ_ground_truth = 70.0 * ground_truth - 10.0
    if dataset == 'Shanghai_2020':
        dBZ_output = 70.0 * output
        dBZ_ground_truth = 70.0 * ground_truth
    if len(output.size()) == 5:  # [seq_len, batch_size, channels=1, height, width]
        dim = [2, 3, 4]
    elif len(output.size()) == 4:  # [seq_len, channels=1, height, width]
        dim = [1, 2, 3]
    elif len(output.size()) == 3:  # [channels=1, height, width]
        dim = [0, 1, 2]
    output_ = (t.ge(dBZ_output, dBZ_downvalue) & t.le(dBZ_output, dBZ_upvalue)).int()
    ground_truth_ = (t.ge(dBZ_ground_truth, dBZ_downvalue) & t.le(dBZ_ground_truth, dBZ_upvalue)).int()
    index = t.eq(t.sum(ground_truth_, dim=dim), 0)  #  find the index where the ground-truth sample has no rainfall preddiction hits

    hits = t.sum(output_ * ground_truth_, dim=dim)
    misses = t.sum(ground_truth_ * (1 - output_), dim=dim)
    false_alarms = t.sum(output_ * (1 - ground_truth_), dim=dim)
    correct_rejections = t.sum((1 - output_) * (1 - ground_truth_), dim=dim)
    pod = hits.float() / (hits + misses).float()
    far = false_alarms.float() / (hits + false_alarms).float()
    csi = hits.float() / (hits + misses + false_alarms).float()
    bias = (hits + false_alarms).float() / (hits + misses).float()
    hss = (2.0 * (hits * correct_rejections - misses * false_alarms)).float() / (
                (hits + misses) * (misses + correct_rejections) + (hits + false_alarms) * (
                    false_alarms + correct_rejections)).float()

    # pod = t.where(t.isnan(pod), t.full_like(pod, 0.0), pod)  # replace the nan in pod to 0  (nan is appeared when hits and misses are both 0)
    far = t.where(t.isnan(far), t.full_like(far, 1.0), far)  # replace the nan in far to 1  (nan is appeared when hits and false alarms are both 0)
    # bias = pod / (1.0 - far)
    # bias = t.where(t.isnan(bias), t.full_like(bias, 0.0), bias)
    # csi = t.where(t.isnan(csi), t.full_like(csi, 0.0), csi)
    # hss = t.where(t.isnan(hss), t.full_like(hss, 0.0), hss)
    pod = t.where(index, t.full_like(pod, 0), pod)
    far = t.where(index, t.full_like(far, 0), far)
    csi = t.where(index, t.full_like(csi, 0), csi)
    hss = t.where(index, t.full_like(hss, 0), hss)
    bias = t.where(index, t.full_like(bias, 0), bias)
    return pod, far, csi, bias, hss, index.int()


def compute_ssim(output, ground_truth):
    if len(output.shape) == 5:  # [seq_len, batch_size, channels=1, height, width]
        ssim_seq = []
        seq_len = output.shape[0]
        batch_size = output.shape[1]
        for i in range(seq_len):
            ssim_batch = []
            for j in range(batch_size):
                ssim = structural_similarity(output[i, j, 0], ground_truth[i, j, 0], data_range=1)
                ssim_batch.append(ssim)
            ssim_seq.append(ssim_batch)
    return t.Tensor(ssim_seq)
