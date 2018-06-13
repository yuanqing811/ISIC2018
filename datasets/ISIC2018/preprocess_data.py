if __name__ == '__main__':

    from datasets.ISIC2018 import load_training_data, resize_and_save_task12

    resize_and_save_task12(output_size=1024)

    # _, _, _ = load_training_data(task_idx=1, output_size=224)
    # _, _, _ = load_training_data(task_idx=2, output_size=224)
    # _, _, _ = load_training_data(task_idx=3, output_size=224)
