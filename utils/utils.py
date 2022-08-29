import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import xlwt
import numpy as np
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt


def ini_model_params(model, ini_params_mode):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
            if ini_params_mode == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif ini_params_mode == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def img_seq_summary(img_seq, global_step, name_scope, writer):
    seq_len = img_seq.size()[0]
    for i in range(seq_len):
        writer.add_images(name_scope + '/Img' + str(i + 1), img_seq[i], global_step)


def save_test_results(log_dir, pod, far, csi, ssim=None):
    test_results_path = os.path.join(log_dir, 'test_results.xls')
    work_book = xlwt.Workbook(encoding='utf-8')
    sheet = work_book.add_sheet('sheet')
    sheet.write(0, 0, 'pod')
    for col, label in enumerate(pod.tolist()):
        sheet.write(0, 1 + col, str(label))
    sheet.write(1, 0, 'far')
    for col, label in enumerate(far.tolist()):
        sheet.write(1, 1 + col, str(label))
    sheet.write(2, 0, 'csi')
    for col, label in enumerate(csi.tolist()):
        sheet.write(2, 1 + col, str(label))
    if ssim is not None:
        sheet.write(5, 0, 'ssim')
        for col, label in enumerate(ssim.tolist()):
            sheet.write(5, 1 + col, str(label))
    work_book.save(test_results_path)


def save_test_samples_imgs(log_dir, index, input, ground_truth, output, dataset='HKO_7', save_mode='integral'):
    if dataset == 'HKO_7':
        input = 70.0 * input - 10.0
        output = 70.0 * output - 10.0
        ground_truth = 70.0 * ground_truth - 10.0
    if dataset == 'Shanghai_2020':
        input = 70.0 * input
        output = 70.0 * output
        ground_truth = 70.0 * ground_truth
    input_seq_len = input.size()[0]
    out_seq_len = output.size()[0]
    height = input.size()[4]
    width = input.size()[3]
    x = np.arange(0, width)
    y = np.arange(height, 0, -1)
    levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    cmp = mpl.colors.ListedColormap(['white', 'lightskyblue', 'cyan', 'lightgreen', 'limegreen', 'green',
                                     'yellow', 'orange', 'chocolate', 'red', 'firebrick', 'darkred', 'fuchsia',
                                     'purple'], 'indexed')
    if not os.path.exists(os.path.join(log_dir, 'sample' + str(index + 1))):
        os.makedirs(os.path.join(log_dir, 'sample' + str(index + 1)))
    for i in range(input_seq_len):
        img = input[i].squeeze().cpu().numpy()
        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'input' + str(i + 1))
        if save_mode == 'simple':
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        elif save_mode == 'integral':
            plt.title('Input')
            plt.xlabel('Timestep' + str(i + 1))
            plt.colorbar()
            plt.savefig(save_fig_path, dpi=600)
        plt.clf()
    for i in range(out_seq_len):
        img = output[i].squeeze().cpu().numpy()
        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'output' + str(i + 1))
        if save_mode == 'simple':
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        elif save_mode == 'integral':
            plt.title('Output')
            plt.xlabel('Timestep' + str(i + 1))
            plt.colorbar()
            plt.savefig(save_fig_path, dpi=600)
        plt.clf()
    for i in range(out_seq_len):
        img = ground_truth[i].squeeze().cpu().numpy()
        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'ground_truth' + str(i + 1))
        if save_mode == 'simple':
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        elif save_mode == 'integral':
            plt.title('Ground_truth')
            plt.xlabel('Timestep' + str(i + 1))
            plt.colorbar()
            plt.savefig(save_fig_path, dpi=600)
        plt.clf()
    return
