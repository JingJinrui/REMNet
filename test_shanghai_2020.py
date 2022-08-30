import os
import utils
import models
import datasets
import torch as t
from torch import nn
from tqdm import tqdm
from configs import config_sh as config
from datetime import datetime
from torch.utils.data import DataLoader


def test(test_samples_only=True, dBZ_threshold=10, eval_ssim=False):
    os.makedirs(config.test_results_save_dir, exist_ok=True)
    test_results_save_dir = os.path.join(config.test_results_save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(test_results_save_dir, exist_ok=True)

    model = models.remnet(in_seq_len=config.in_seq_len, out_seq_len=config.out_seq_len, use_gpu=config.use_gpu)
    model.load_state_dict(t.load('{}/{}.pth'.format(config.checkpoints_dir, config.model_name)))
    if config.use_gpu:
        device = t.device('cuda')
        if len(config.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=config.device_ids, dim=1)
            model.to(device)
        else:
            model.to(device)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load Pretrained Model Successfully")

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Test Model on Test Samples")
    test_samples_dataset = datasets.Shanghai_2020_samples(config.test_samples_dir)
    test_samples_loader = DataLoader(test_samples_dataset, batch_size=1, shuffle=False)
    with t.no_grad():
        for iter, data in tqdm(enumerate(test_samples_loader)):
            input, ground_truth = t.split(data.permute(1, 0, 2, 3, 4), [config.in_seq_len, config.out_seq_len], dim=0)
            if config.use_gpu:
                device = t.device('cuda')
                input = input.to(device)
                ground_truth = ground_truth.to(device)
            output = model(input)
            utils.save_test_samples_imgs(test_results_save_dir, test_samples_dataset.test_samples[iter], input,
                                         ground_truth, output, config.dataset,
                                         save_mode='integral')

    if not test_samples_only:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Test Model on Test Dataset")
        test_dataset = datasets.Shanghai_2020(config.dataset_root, seq_len=config.seq_len,
                                              seq_interval=config.seq_interval,
                                              train=False,
                                              test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, config.test_batch_size, shuffle=False, num_workers=config.num_workers)
        pod, far, csi, ssim = utils.valid(model, test_dataloader, config, dBZ_threshold=dBZ_threshold,
                                               eval_by_seq=True, eval_ssim=eval_ssim)
        if eval_ssim:
            print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, SSIM: {:.4f}'
                  .format(t.mean(pod), t.mean(far), t.mean(csi), t.mean(ssim)))
        else:
            print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}'
                  .format(t.mean(pod), t.mean(far), t.mean(csi)))
        utils.save_test_results(test_results_save_dir, pod, far, csi, ssim)


if __name__ == '__main__':
    test()
