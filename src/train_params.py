import os
import datetime


# model pretrain weights
unet_pretrain_weight = ''
resnet50_pretrain_weight = ''

# log dir
now = datetime.datetime.now()
time_str = 'train-'+now.strftime("[%Y-%m-%d]-[%H:%M]")
output_path='' 
output_path_time = os.path.join(output_path,time_str)
if not os.path.exists(output_path_time):
    os.mkdir(output_path_time)
# traning intermediate results
tensorboard_log_path = os.path.join(output_path_time, 'tensorboard_log')
log_path = os.path.join(output_path_time, 'run_logs.log')


# data loader
s1_video_clip_dir =''
s2_video_clip_dir =''
s3_video_clip_dir =''
train_identity_file = ''
train_emotion_file = ''
test_identity_file = ''
test_emotion_file = ''
num_frames = 16
image_size = 256

# training 
num_epochs = 300
batch_size = 16
num_workers = 8
learning_rate = 1e-4
gpu_ids = [0,1,2,3]




# the paths of best models at stage 1
best_f_hpr = ''
best_f_lpr = ''


# paths for saving stage 1 intermediate results
training_batch_tensor_path = ""
validation_batch_tensor_path = ""



# stage 2
# model weights
uvit_256_weight = ''
dfew_image_fer_weight = ''

# s2 training 
s2_learning_rate = 1e-4
s2_batch_size = 16








