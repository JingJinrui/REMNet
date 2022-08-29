import os
import utils
import models
import datasets
import torch as t
import models.discriminators as dis
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from configs import config_sh as config
from torch.utils.tensorboard import SummaryWriter


def train():
    os.makedirs(config.log_dir, exist_ok=True)
    log_dir = os.path.join(config.log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    log_path = open(os.path.join(log_dir, 'log.txt'), 'w')
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # define the model
    model = models.remnet(in_seq_len=config.in_seq_len, out_seq_len=config.out_seq_len, use_gpu=config.use_gpu)
    seq_d = dis.remnet_sequence_discriminator()
    fra_d = dis.remnet_frame_patch_discriminator()
    resume_epoch = 0
    utils.ini_model_params(model, config.ini_params_mode)
    utils.ini_model_params(seq_d, config.ini_params_mode)
    utils.ini_model_params(fra_d, config.ini_params_mode)
    print(
        "Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Initialize the Model Parameters Use {}".format(
            config.ini_params_mode))
    print(
        "Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Initialize the Model Parameters Use {}".format(
            config.ini_params_mode), file=log_path)

    # train with gpu if configs.use_gpu is True, parallel the train if there has multiple gpus
    if config.use_gpu:
        device = t.device('cuda')
        if len(config.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=config.device_ids, dim=1)
        model.to(device)
        seq_d.to(device)
        fra_d.to(device)
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Distribute the Train on {} GPUs".format(
            len(config.device_ids)))
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Distribute the Train on {} GPUs".format(
            len(config.device_ids)), file=log_path)

    train_dataset = datasets.Shanghai_2020(config.dataset_root, seq_len=config.seq_len, seq_interval=config.seq_interval,
                                   train=True,
                                   test=False, nonzero_points_threshold=None)
    train_dataloader = DataLoader(train_dataset, config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    valid_dataset = datasets.Shanghai_2020(config.dataset_root, seq_len=config.seq_len, seq_interval=config.seq_interval,
                                   train=False,
                                   test=False, nonzero_points_threshold=None)
    valid_dataloader = DataLoader(valid_dataset, config.valid_batch_size, shuffle=False, num_workers=config.num_workers)
    iters_per_epoch = train_dataloader.__len__()
    iters = resume_epoch * iters_per_epoch

    # define loss functions
    criterion1 = utils.weighted_l1_loss
    lam1 = 1.0
    criterion2 = utils.mixed_adversarial_loss
    lam2 = 0.05
    criterion3 = utils.perceptual_similarity_loss
    lam3 = 2.0
    seq_d_criterion = utils.seq_d_bce_adversarial_loss
    fra_d_criterion = utils.fra_d_hinge_adversarial_loss

    optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.optim_betas)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=config.scheduler_gamma)
    seq_d_optimizer = t.optim.Adam(seq_d.parameters(), lr=config.discriminator_learning_rate,
                                   betas=config.discriminator_optim_betas)
    seq_d_scheduler = t.optim.lr_scheduler.StepLR(seq_d_optimizer, step_size=10,
                                                  gamma=config.discriminator_scheduler_gamma)
    fra_d_optimizer = t.optim.Adam(fra_d.parameters(), lr=config.discriminator_learning_rate,
                                   betas=config.discriminator_optim_betas)
    fra_d_scheduler = t.optim.lr_scheduler.StepLR(fra_d_optimizer, step_size=10,
                                                  gamma=config.discriminator_scheduler_gamma)

    best_csi = 0
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train")
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train", file=log_path)

    for epoch in range(resume_epoch, config.train_max_epochs):
        for iter, data in enumerate(train_dataloader):
            input, ground_truth = t.split(data.permute(1, 0, 2, 3, 4), [config.in_seq_len, config.out_seq_len], dim=0)
            if config.use_gpu:
                input = input.to(device)
                ground_truth = ground_truth.to(device)
            output = model(input)

            if (iter + 1) % config.discriminator_train_fre == 0:
                fra_d_optimizer.zero_grad()
                fra_d_loss_real, fra_d_loss_fake = fra_d_criterion(output.detach(), ground_truth, fra_d)
                fra_d_loss = fra_d_loss_real + fra_d_loss_fake
                fra_d_loss.backward()
                fra_d_optimizer.step()

                seq_d_optimizer.zero_grad()
                seq_d_loss_real, seq_d_loss_fake = seq_d_criterion(input, output.detach(), ground_truth, seq_d)
                seq_d_loss = seq_d_loss_real + seq_d_loss_fake
                seq_d_loss.backward()
                seq_d_optimizer.step()

            if (iter + 1) % config.model_train_fre == 0:
                optimizer.zero_grad()
                loss1 = criterion1(output, ground_truth, config.dataset)
                loss2_seq_d, loss2_fra_d = criterion2(input, output, seq_d, fra_d)
                loss2 = loss2_seq_d + loss2_fra_d
                loss3 = criterion3(output, ground_truth, model.module.frame_encoder, randomly_sampling=10)
                loss = lam1 * loss1 + lam2 * loss2 + lam3 * loss3
                loss.backward()
                optimizer.step()

            iters += 1

            if (iter + 1) % config.loss_log_iters == 0:
                print('Time: ' + datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tseq_d_Loss: {:.6f}\tfra_d_Loss: {:.6f}'.format(
                    (epoch + 1), (iter + 1) * config.train_batch_size, len(train_dataset),
                                 100. * (iter + 1) / iters_per_epoch, loss.item(), seq_d_loss.item(),
                    fra_d_loss.item()))
                writer.add_scalar('Train/Loss/loss', loss.item(), iters)
                writer.add_scalar('Train/Loss/loss/loss1', loss1.item(), iters)
                writer.add_scalar('Train/Loss/loss/loss2_seq_d', loss2_seq_d.item(), iters)
                writer.add_scalar('Train/Loss/loss/loss2_fra_d', loss2_fra_d.item(), iters)
                writer.add_scalar('Train/Loss/loss/loss3', loss3.item(), iters)
                writer.add_scalar('Train/Loss/seq_d_loss_real', seq_d_loss_real.item(), iters)
                writer.add_scalar('Train/Loss/seq_d_loss_fake', seq_d_loss_fake.item(), iters)
                writer.add_scalar('Train/Loss/fra_d_loss_real', fra_d_loss_real.item(), iters)
                writer.add_scalar('Train/Loss/fra_d_loss_fake', fra_d_loss_fake.item(), iters)

            if (iter + 1) % config.img_log_iters == 0:
                utils.img_seq_summary(ground_truth, iters, 'Train/Imgs/Ground Truth', writer)
                utils.img_seq_summary(output, iters, 'Train/Imgs/Prediction', writer)

        scheduler.step()
        seq_d_scheduler.step()
        fra_d_scheduler.step()

        pod, far, csi, _ = utils.valid(model, valid_dataloader, config, dBZ_threshold=10, eval_ssim=False)
        print('Time: ' + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\tPOD: {:.4f}\tFAR: {:.4f}\tCSI: {:.4f}'.format(
            (epoch + 1), pod, far, csi))
        print('Time: ' + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\tPOD: {:.4f}\tFAR: {:.4f}\tCSI: {:.4f}'.format(
            (epoch + 1), pod, far, csi), file=log_path)
        writer.add_scalar('Valid/POD', pod, epoch + 1)
        writer.add_scalar('Valid/FAR', far, epoch + 1)
        writer.add_scalar('Valid/CSI', csi, epoch + 1)

        if csi > best_csi:
            best_csi = csi
            print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\t  Save the Current Best Model'.format(epoch + 1))
            print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\t  Save the Current Best Model'.format(epoch + 1),
                    file=log_path)
            if len(config.device_ids) > 1:
                t.save(model.module.state_dict(), '{}/{}.pth'.format(config.checkpoints_dir, config.model_name))
            else:
                t.save(model.state_dict(), '{}/{}.pth'.format(config.checkpoints_dir, config.model_name))

        if (epoch + 1) % config.model_save_fre == 0:
            if len(config.device_ids) > 1:
                t.save(model.module.state_dict(),
                       '{}/{}_epoch_{}.pth'.format(config.checkpoints_dir, config.model_name, (epoch + 1)))
            else:
                t.save(model.state_dict(),
                       '{}/{}_epoch_{}.pth'.format(config.checkpoints_dir, config.model_name, (epoch + 1)))

    log_path.close()
    writer.close()


if __name__ == '__main__':
    train()
