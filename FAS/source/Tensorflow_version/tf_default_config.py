# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : tf_default_config.py
# @Software : PyCharm

import tensorflow as tf
from datetime import datetime
import os


def get_default_config():
    conf = {}

    # ----------------------training---------------
    conf['lr'] = 1e-1
    conf['milestones'] = [10, 15, 22]
    conf['gamma'] = 0.1
    conf['epochs'] = 25
    conf['momentum'] = 0.9
    conf['batch_size'] = 1024

    # model
    conf['num_classes'] = 3
    conf['input_channel'] = 3
    conf['embedding_size'] = 128

    # dataset
    conf['train_root_path'] = './datasets/rgb_image'

    # save file path
    conf['snapshot_dir_path'] = './saved_logs/snapshot'

    # log path
    conf['log_path'] = './saved_logs/jobs'
    
    # TensorBoard
    conf['board_loss_every'] = 10
    
    # save model/iter
    conf['save_every'] = 30

    return conf


def update_config(args, conf):
    conf['devices'] = args.devices
    conf['patch_info'] = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf['input_size'] = [h_input, w_input]
    
    # ✅ Dùng adaptive kernel_size giống PyTorch
    conf['kernel_size'] = get_kernel(h_input, w_input)   # (16, 16) cho 256x256
    
    conf['device'] = "/GPU:{}".format(conf['devices'][0]) if tf.config.list_physical_devices('GPU') else "/CPU:0"

    conf['ft_height'] = 2 * conf['kernel_size'][0]
    conf['ft_width'] = 2 * conf['kernel_size'][1]


    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf['log_path'], job_name, current_time)
    snapshot_dir = '{}/{}'.format(conf['snapshot_dir_path'], job_name)

    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    conf['model_path'] = snapshot_dir
    conf['log_path'] = log_path
    conf['job_name'] = job_name
    return conf


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size