import torch as t
from skimage.metrics import structural_similarity


def crosstab_evaluate(output, ground_truth, dBZ_downvalue, dBZ_upvalue, dataset):
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

    # pod = t.where(t.isnan(pod), t.full_like(pod, 0.0), pod)  # replace the nan in pod to 0  (nan is appeared when hits and misses are both 0)
    far = t.where(t.isnan(far), t.full_like(far, 1.0), far)  # replace the nan in far to 1  (nan is appeared when hits and false alarms are both 0)
    # csi = t.where(t.isnan(csi), t.full_like(csi, 0.0), csi)
    pod = t.where(index, t.full_like(pod, 0), pod)
    far = t.where(index, t.full_like(far, 0), far)
    csi = t.where(index, t.full_like(csi, 0), csi)
    return pod, far, csi, index.int()


def calculate_ssim(output, ground_truth):
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


def valid(model, dataloader, config, dBZ_threshold=10, eval_by_seq=False, eval_ssim=False):
    model.eval()
    pod = []
    far = []
    csi = []
    index = []
    ssim = []
    with t.no_grad():
        for iter, data in enumerate(dataloader):
            input, ground_truth = t.split(data.permute(1, 0, 2, 3, 4), [config.in_seq_len, config.out_seq_len], dim=0)
            if config.use_gpu:
                device = t.device('cuda')
                input = input.to(device)
                ground_truth = ground_truth.to(device)
            output = model(input)
            pod_, far_, csi_, index_ = crosstab_evaluate(output, ground_truth, dBZ_threshold, 70, config.dataset)
            pod.append(pod_.data)
            far.append(far_.data)
            csi.append(csi_.data)
            index.append(index_)
            if eval_ssim:
                ssim_ = calculate_ssim(output.cpu().numpy(), ground_truth.cpu().numpy())
                ssim.append(ssim_)
        index = t.cat(index, dim=1)
        data_num = index.numel()
        # the ground-truth sample which has no rainfall preddiction hits will not be included in calculation
        cal_num = index.size()[1] - t.sum(index, dim=1) if eval_by_seq is True else data_num - t.sum(index)
        pod = t.sum(t.cat(pod, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(t.cat(pod, dim=1)) / cal_num
        far = t.sum(t.cat(far, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(t.cat(far, dim=1)) / cal_num
        csi = t.sum(t.cat(csi, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(t.cat(csi, dim=1)) / cal_num
        if eval_ssim:
            ssim = t.mean(t.cat(ssim, dim=1), dim=1) if eval_by_seq is True else t.mean(t.cat(ssim, dim=1))
        else:
            ssim = None
    model.train()
    return pod, far, csi, ssim
