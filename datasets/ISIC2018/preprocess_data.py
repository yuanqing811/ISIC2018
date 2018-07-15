if __name__ == '__main__':

    from datasets.ISIC2018 import load_training_data, load_validation_data, load_test_data

    _, _, _ = load_training_data(task_idx=1, output_size=224)
    _, _, _ = load_training_data(task_idx=2, output_size=224)
    _, _, _ = load_training_data(task_idx=3, output_size=224)

    images, image_names, image_sizes = load_validation_data(task_idx=1, output_size=224)
    images, image_names = load_validation_data(task_idx=3, output_size=224)

    images, image_names, image_sizes = load_test_data(task_idx=1, output_size=224)
    images, image_names = load_test_data(task_idx=3, output_size=224)

