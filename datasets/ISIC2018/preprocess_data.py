if __name__ == '__main__':

    from datasets.ISIC2018 import load_training_data
    _, _, _ = load_training_data(task_idx=1, output_size=224)
    _, _, _ = load_training_data(task_idx=2, output_size=224)
    _, _, _ = load_training_data(task_idx=3, output_size=224)

