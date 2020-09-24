import os
import tensorflow as tf
import numpy as np

home = "F:\\Masterarbeit\\DLR\project\\1_cnn_truck_detection\\training_data\\tf"


def create_archive(home):
    os.mkdir(os.path.join(home, "data"))
    model_dir = os.path.join(home, "model")
    os.mkdir(model_dir)
    os.mkdir(os.path.join(model_dir, "my_model"))