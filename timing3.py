import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import glob
import time
import numpy as np
import random
from PIL import Image

#imports from raspberi pi
#import RPi.GPIO as GPIO
#GPIO.setmode(GPIO.BOARD )
#GPIO.setup(18, GPIO.OUT)
#GPIO.output(18, GPIO.LOW)

#imports for jettson nano
#import Jetson.GPIO as GPIO
#channel=
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(channel, GPIO.OUT)
#GPIO.output(channel, GPIO.LOW)

begin=0
end=0
diff=0


def number_of_steps(data_size,batch_size):
    steps=int(data_size/batch_size)
    return steps

def prepare_image(file_path,batch_size):
    file_list = glob.glob(validation_data_folder + '*.jpg')
    file_list.extend(glob.glob(validation_data_folder + '*.jpeg'))
    file_list.extend(glob.glob(validation_data_folder + '*.png'))

    length=len(file_list)
    if batch_size <= 0 or batch_size >=length:
        print("incorrect batch size")
        return

    if batch_size==1:
        choice=random.randint(0,length-1)
        file=file_list[choice]
        im=Image.open(file)
        im2=im.resize((112, 112))
        im3=np.expand_dims((np.array(im2)/255),axis=0)
        return im3

    choose_list=[int]*length

    for i in range(length):
        choose_list[i]=i

    choose_list=np.array(choose_list)
    np.random.shuffle(choose_list)
    batch_img=[]
    for j in range(batch_size):
        file = file_list[choose_list[j]]
        im = Image.open(file)
        im2 = im.resize((112, 112))
        im3 = (np.array(im2) / 255)
        batch_img+=[im3]
    batch_img=np.array(batch_img)
    return batch_img



def start_time():
    global begin
    begin=time.time()
    #for raspberi pi
    #GPIO.output(18, GPIO.HIGH)

    #for jettson nano
    #GPIO.output(channel, GPIO.HIGH)

def end_time():
    global bein,end,diff
    end=time.time()
    diff=end-begin
    #for raspberi pi
    #GPIO.output(18, GPIO.LOW)

    #for jettson nano
    #GPIO.output(channel, GPIO.HIGH)


def calculate_iou(target_boxes, pred_boxes):
    xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
    yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
    xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
    yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
    interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
    boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

single_batch_time=0
sixteen_batch_time=0
thirty_two_batch_time=0
sixty_four_batch_time=0


def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    return mse + (1 - iou)

def iou_metric(y_true, y_pred):
    return calculate_iou(y_true, y_pred)

##############################################################################################
mob='mobileSmaller.h5'
res='resnetSmaller.h5'

model_source=res

validation_data_folder='/home/pi/Desktop/project/validation data/'
#validation_data_folder='E:/machine learning/augval/'
file_list = glob.glob(validation_data_folder + '*.jpg')
file_list.extend(glob.glob(validation_data_folder + '*.jpeg'))
file_list.extend(glob.glob(validation_data_folder + '*.png'))
data_size=len(file_list)
model = tf.keras.models.load_model(model_source,
                                       custom_objects={'custom_loss': custom_loss, 'iou_metric': iou_metric,
                                                       'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.3)},compile=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=custom_loss,
    metrics=[iou_metric]
)

i_1=prepare_image(validation_data_folder,1)
i_16=prepare_image(validation_data_folder,16)
i_32=prepare_image(validation_data_folder,32)
i_64=prepare_image(validation_data_folder,64)
i_640=prepare_image(validation_data_folder,640)

time_arr = []
print("testing one images at a time")
for i in range(10):
    temp = i_640[i]
    temp2 = np.expand_dims(temp, axis=0)
    start_time()
    pred = model.predict(temp2)
    end_time()
    time_arr += [diff]
time_arr = np.array(time_arr)
avg = np.average(time_arr[1:len(time_arr)])
fps = 1 / avg
print("average time of single one by one image inference is ")
print(str(avg), " seconds")
print("which translated to ", str(fps), " frames per second performance")
# input("continue")
print("")

time_arr = []
print("testing sending 2 images at a time")
for i in range(10):
    temp = i_640[i:i + 2]
    start_time()
    pred = model.predict(temp2, batch_size=2)
    end_time()
    time_arr += [diff]
time_arr = np.array(time_arr)
avg = np.average(time_arr)
fps = 2 / avg
print("average time of 2 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ", str(fps), " frames per second performance")
# input("continue")
print("")

time_arr = []
print("testing sending 4 images at a time")
for i in range(10):
    temp = i_640[i:i + 4]
    start_time()
    pred = model.predict(temp2, batch_size=4)
    end_time()
    time_arr += [diff]
time_arr = np.array(time_arr)
avg = np.average(time_arr)
fps = 4 / avg
print("average time of 4 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ", str(fps), " frames per second performance")
# input("continue")
print("")

time_arr = []
print("testing sending 8 images at a time")
for i in range(10):
    temp = i_640[i:i + 8]
    start_time()
    pred = model.predict(temp2, batch_size=8)
    end_time()
    time_arr += [diff]
time_arr = np.array(time_arr)
avg = np.average(time_arr)
fps = 8 / avg
print("average time of 8 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ", str(fps), " frames per second performance")
# input("continue")
print("")

time_arr = []
print("testing sending 16 files at a time")
for i in range(10):
    temp = i_640[i:i + 16]
    start_time()
    pred = model.predict(temp2, batch_size=16)
    end_time()
    time_arr += [diff]
time_arr = np.array(time_arr)
avg = np.average(time_arr)
fps = 16 / avg
print("average time of 16 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ", str(fps), " frames per second performance")
# input("continue")
print("")

time_arr = []
print("testing sending 32 files at a time")
for i in range(32):
    temp = i_640[i:i + 32]
    start_time()
    pred = model.predict(temp2, batch_size=32)
    end_time()
    time_arr += [diff]
time_arr = np.array(time_arr)
avg = np.average(time_arr)
fps = 32 / avg
print("average time of 32 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ", str(fps), " frames per second performance")
# input("continue")
print("")