import math
import random
import torch as t
import torch.nn.functional as F


def weighted_l1_loss(output, ground_truth, dataset='HKO_7'):
    if dataset == 'HKO_7':
        dBZ_ground_truth = 70.0 * ground_truth - 10.0
        weight_matrix = t.clamp(t.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0)
    elif dataset == 'Shanghai_2020':
        dBZ_ground_truth = 70.0 * ground_truth
        weight_matrix = t.clamp(t.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0)
    return t.mean(weight_matrix * t.abs(output - ground_truth))


def weighted_l2_loss(output, ground_truth, dataset='HKO_7'):
    if dataset == 'HKO_7':
        dBZ_ground_truth = 70.0 * ground_truth - 10.0
        weight_matrix = t.clamp(t.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0)
    elif dataset == 'Shanghai_2020':
        dBZ_ground_truth = 70.0 * ground_truth
        weight_matrix = t.clamp(t.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0)
    return t.mean(weight_matrix * t.pow(t.abs(output - ground_truth), 2.0))


def perceptual_similarity_loss(output, ground_truth, encoder, randomly_sampling=None):
    seq_len = output.size()[0]
    if randomly_sampling is not None:
        index = random.sample(range(0, seq_len), randomly_sampling)
        output_feature = encoder(output[index])
        ground_truth_feature = encoder(ground_truth[index])
    else:
        output_feature = encoder(output)
        ground_truth_feature = encoder(ground_truth)
    return t.mean(t.pow(t.abs(output_feature - ground_truth_feature), 2.0))


def seq_d_hinge_adversarial_loss(input, output, ground_truth, seq_d):
    real_seq = seq_d(t.cat([input, ground_truth], dim=0))
    fake_seq = seq_d(t.cat([input, output], dim=0))
    seq_d_loss_real = t.mean(t.relu(1.0 - real_seq))
    seq_d_loss_fake = t.mean(t.relu(1.0 + fake_seq))
    return seq_d_loss_real, seq_d_loss_fake


def fra_d_hinge_adversarial_loss(output, ground_truth, fra_d):
    real_fra = fra_d(ground_truth)
    fake_fra = fra_d(output)
    fra_d_loss_real = t.mean(t.relu(1.0 - real_fra))
    fra_d_loss_fake = t.mean(t.relu(1.0 + fake_fra))
    return fra_d_loss_real, fra_d_loss_fake


def hinge_adversarial_loss(input, output, seq_d, fra_d):
    fake_seq = seq_d(t.cat([input, output], dim=0))
    fake_fra = fra_d(output)
    g_loss_seq = -t.mean(fake_seq)
    g_loss_fra = -t.mean(fake_fra)
    return g_loss_seq, g_loss_fra


def seq_d_bce_adversarial_loss(input, output, ground_truth, seq_d):
    real_seq = seq_d(t.cat([input, ground_truth], dim=0))
    fake_seq = seq_d(t.cat([input, output], dim=0))
    seq_d_loss_real = F.binary_cross_entropy(real_seq, t.ones_like(real_seq))
    seq_d_loss_fake = F.binary_cross_entropy(fake_seq, t.zeros_like(fake_seq))
    return seq_d_loss_real, seq_d_loss_fake


def fra_d_bce_adversarial_loss(output, ground_truth, fra_d):
    real_fra = fra_d(ground_truth)
    fake_fra = fra_d(output)
    fra_d_loss_real = F.binary_cross_entropy(real_fra, t.ones_like(real_fra))
    fra_d_loss_fake = F.binary_cross_entropy(fake_fra, t.zeros_like(fake_fra))
    return fra_d_loss_real, fra_d_loss_fake


def bce_adversarial_loss(input, output, seq_d, fra_d):
    fake_seq = seq_d(t.cat([input, output], dim=0))
    fake_fra = fra_d(output)
    g_loss_seq = F.binary_cross_entropy(fake_seq, t.ones_like(fake_seq))
    g_loss_fra = F.binary_cross_entropy(fake_fra, t.ones_like(fake_fra))
    return g_loss_seq, g_loss_fra


def mixed_adversarial_loss(input, output, seq_d, fra_d):
    fake_seq = seq_d(t.cat([input, output], dim=0))
    fake_fra = fra_d(output)
    g_loss_seq = F.binary_cross_entropy(fake_seq, t.ones_like(fake_seq))
    g_loss_fra = -t.mean(fake_fra)
    return g_loss_seq, g_loss_fra
