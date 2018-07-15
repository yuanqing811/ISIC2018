import os
import inspect


def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))
model_data_dir = os.path.join(root_dir, 'model_data')
submission_dir = os.path.join(root_dir, 'submissions')

dir_to_make = [model_data_dir, submission_dir]
mkdir_if_not_exist(dir_list=dir_to_make)
mkdir_if_not_exist(dir_list=dir_to_make)
