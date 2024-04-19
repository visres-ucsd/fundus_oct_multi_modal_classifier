# file to store all code constants for training

# file paths....
modality_type = "oct"
project_dir = "/tscc/nfs/home/vejoshi/oct_fundus_project/"
dataset_dir    = project_dir + "oct_fundus_dataset/"
dataset_path   = dataset_dir + "fundus_images/" if modality_type == "fundus" else dataset_dir+"oct_images/"
label_path     = dataset_dir + "fundus_labels_binary_small_version.pickle" if modality_type == "fundus" else dataset_dir + "oct_labels_binary_small_version.pickle"
test_ids_paths = dataset_dir + "test_patient_ids_fundus_oct.pickle"

# training constants
use_aug = True
training_nature = "supervised_only"
model_name = "resnet"
input_shape = (224,224,3)
unfreeze_perc = 1.0
frozen_epochs = 10 # keep the base model weights frozen for these many epochs otherwise the classification heads would damage the
pre_freeze_lr = 1e-02
learning_rate = 1e-03
dropout = 0.4
focal_weight = 4.0
l2_reg = 1e-02
pool_type = "max"
dense_1 = 8
dense_2 = 12
dense_3 = 24
batch_size = 32
decision_threshold = 0.5 # used by metrics
train_epochs = 100
num_train_samples_viz = 4
patience = 10
reduce_lr_patience = 3
lr_scale = 0.1
lab_smooth = 0.13
aug_prob = 0.9

# Label constants...
"""
label_mapping = {"healthy"  : [1,0,0],
                 "suspects" : [0,1,0],
                 "glaucoma" : [0,0,1]}
"""
label_mapping = {"glaucoma" : 0,
                 "healthy"  : 1}

num_classes = len(label_mapping)

# Save directory ##########################################################
# Creating directory to save runs & best weights.....
save_dir_name = "./expeiments/" + modality_type + "/"
model_save_name = model_name + "_shape_" + str(input_shape[0]) + "_pool_" + str(pool_type) + "_dp_" + \
str(dropout) + "_lr_" + str(learning_rate) + "_dense_" + str(dense_1) + "_" + str(dense_2) + "_" + str(dense_3)


