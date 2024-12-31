import datetime
import os


clip_dir = ''

training_label = ''
test_label = ''

num_frames = 16

num_classes = 7

batch_size = 32

num_workers = 8

num_epochs = 200

learning_rate = 1e-5

image_size = 224 

dfew_emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

dfew_emotion_classes_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

dfew_emotion_classes_dict_reverse = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

gpu_ids = [0, 1, 2, 3]

pretrain_weights_dir = ''



# model saving path
now = datetime.datetime.now()
time_str = 'pretrain-'+now.strftime("[%Y-%m-%d]-[%H:%M]")

# The output path is the folder where the model will be saved
# TODO
output_path='' 
output_path_time = os.path.join(output_path,time_str)
if not os.path.exists(output_path_time):
    os.mkdir(output_path_time)
# traning intermediate results
tensorboard_log_path = os.path.join(output_path_time, 'tensorboard_log')
log_path = os.path.join(output_path_time, 'run_logs.log')
