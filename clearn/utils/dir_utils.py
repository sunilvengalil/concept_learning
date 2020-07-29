import os


def check_and_create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_eval_result_dir(result_path, epoch=0, idx=0):
    orig_dir = check_and_create_folder(
        result_path + "/"
        + "reconstructed" + '_{:02d}_{:04}/'.format(epoch, idx))
    return orig_dir
