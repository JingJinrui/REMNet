class Config_HKO_7(object):
    model_name = 'remnet_hko_7'
    ini_params_mode = 'orthogonal'

    dataset = 'HKO_7'
    dataset_root = 'D:/datasets/HKO_7'
    in_seq_len = 5
    out_seq_len = 30
    seq_len = in_seq_len + out_seq_len
    seq_interval = 5

    use_gpu = True
    device_ids = [0, 1, 2, 3]
    num_workers = 8
    train_batch_size = 8
    valid_batch_size = 8
    test_batch_size = 8
    train_max_epochs = 20
    learning_rate = 1e-4
    optim_betas = (0.5, 0.999)
    scheduler_gamma = 0.5
    # adv train config
    discriminator_learning_rate = 1e-4
    discriminator_optim_betas = (0.5, 0.999)
    discriminator_scheduler_gamma = 0.5
    model_train_fre = 1
    discriminator_train_fre = 2

    loss_log_iters = 100
    img_log_iters = 1000
    model_save_fre = 5
    log_dir = './logdir_hko_7'
    checkpoints_dir = './checkpoints_hko_7'
    test_samples_dir = './test_samples_hko_7'
    test_results_save_dir = './test_results_hko_7'


class Config_Shanghai_2020(object):
    model_name = 'remnet_shanghai_2020'
    ini_params_mode = 'orthogonal'

    dataset = 'Shanghai_2020'
    dataset_root = 'D:/datasets/Shanghai_2020'
    in_seq_len = 10
    out_seq_len = 10
    seq_len = in_seq_len + out_seq_len
    seq_interval = None

    use_gpu = True
    device_ids = [0, 1, 2, 3]
    num_workers = 8
    train_batch_size = 8
    valid_batch_size = 8
    test_batch_size = 8
    train_max_epochs = 20
    learning_rate = 1e-4
    optim_betas = (0.5, 0.999)
    scheduler_gamma = 0.5
    # adv train config
    discriminator_learning_rate = 1e-4
    discriminator_optim_betas = (0.5, 0.999)
    discriminator_scheduler_gamma = 0.5
    model_train_fre = 1
    discriminator_train_fre = 2

    loss_log_iters = 100
    img_log_iters = 1000
    model_save_fre = 5
    log_dir = './logdir_shanghai_2020'
    checkpoints_dir = './checkpoints_shanghai_2020'
    test_samples_dir = './test_samples_shanghai_2020'
    test_results_save_dir = './test_results_shanghai_2020'


config_hko = Config_HKO_7()
config_sh = Config_Shanghai_2020()
