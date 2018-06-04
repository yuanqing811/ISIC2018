import os
import errno
from paths import model_data_dir


def get_run_dir(run_name):
    dirname = os.path.join(model_data_dir, run_name)
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    return dirname


def get_weights_filename(run_name):
    dirname = get_run_dir(run_name)
    weights_filename = os.path.join(dirname, '%s.hdf5' % run_name)
    return weights_filename


def get_csv_filename(run_name):
    dirname = get_run_dir(run_name)
    csv_filename = os.path.join(dirname, '%s.csv' % run_name)
    return csv_filename


def get_log_filename(run_name):
    dirname = get_run_dir(run_name)
    log_filename = os.path.join(dirname, '%s_log.txt' % run_name)
    return log_filename


def get_model_summary_filename(run_name):
    dirname = get_run_dir(run_name)
    filename = os.path.join(dirname, '%s.txt' % run_name)
    return filename


def get_model_image_filename(run_name):
    dirname = get_run_dir(run_name)
    filename = os.path.join(dirname, '%s.png' % run_name)
    return filename


def get_json_filename(run_name):
    dirname = get_run_dir(run_name)
    json_filename = os.path.join(dirname, '%s.json' % run_name)
    return json_filename


def get_model_config_filename(run_name):
    dirname = get_run_dir(run_name)
    config_filename = os.path.join(dirname, '%s.pkl' % run_name)
    return config_filename


