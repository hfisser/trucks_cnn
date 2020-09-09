import os

main_dir = os.path.join("F:\\Masterarbeit\\DLR\\project\\1_cnn_truck_detection")

def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)

def setup(main_dir):
    main_tf = os.path.join(main_dir, "TensorFlow")
    create_dir(main_tf)
    tf_addons = os.path.join(main_tf, "addons")
    tf_models = os.path.join(main_tf, "models")
    tf_workspace = os.path.join(main_tf, "workspace")
    tf_training = os.path.join(tf_workspace, "training")
    create_dir(tf_addons)
    create_dir(tf_models)
    create_dir(tf_workspace)
    create_dir(tf_training)
    tf_models_sub = ["community", "official", "orbit", "reserach"]
    for d in tf_models_sub:
        create_dir(os.path.join(tf_models, d))
    tf_training_sub = ["annotations", "exported-models", "images", "models"]
    for d in tf_training_sub:
        create_dir(os.path.join(tf_training, d))
    for d in ["validate", "train"]:
        create_dir(os.path.join(tf_training, "images", d))

if __name__ == "__main__":
    setup(main_dir)
