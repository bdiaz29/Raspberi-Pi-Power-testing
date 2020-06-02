import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
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
        exit(1)
        return

    if batch_size==1:
        choice=random.randint(0,length-1)
        file=file_list[choice]
        im=Image.open(file)
        im2=im.resize((112, 112))
        im3=np.expand_dims((np.array(im2)/255),axis=0).astype(np.float32)
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

def c_iou(target_boxes, pred_boxes):
    xA = max(target_boxes[0], pred_boxes[0])
    yA = max(target_boxes[1], pred_boxes[1])
    xB = max(target_boxes[2], pred_boxes[2])
    yB = max(target_boxes[3], pred_boxes[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (target_boxes[ 2] - target_boxes[0]) * (target_boxes[3] - target_boxes[ 1])
    boxBArea = (pred_boxes[ 2] - pred_boxes[ 0]) * (pred_boxes[3] - pred_boxes[ 1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_iou(a, b, epsilon=1e-5):

    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
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


mob='boundsmobile.h5'
res='boundsresnet50.h5'

model_source=mob
validation_data_folder='/home/pi/Desktop/project/validation data/'
print(str(validation_data_folder))
#validation_data_folder='/home/bruno/Desktop/project/validation data/'
file_list = glob.glob(validation_data_folder + '*.jpg')
file_list.extend(glob.glob(validation_data_folder + '*.jpeg'))
file_list.extend(glob.glob(validation_data_folder + '*.png'))
data_size=len(file_list)


i_1=prepare_image(validation_data_folder,1)
i_16=prepare_image(validation_data_folder,16)
i_32=prepare_image(validation_data_folder,32)
i_64=prepare_image(validation_data_folder,64)
i_640=prepare_image(validation_data_folder,320)

res='resnetsmaller.tflite'
mob='mobilesmaller.tflite'

interpreter = tf.lite.Interpreter(model_path=str(res))
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]






time_arr=[]
print("testing one single file at a time")
for i in range(32):
    temp=i_640[i]
    temp2=np.expand_dims(temp,axis=0).astype(np.float32)
    start_time()
    interpreter.set_tensor(input_index, temp2)
    interpreter.invoke()
    pred_bird = interpreter.get_tensor(output_index)
    end_time()
    time_arr+=[diff]
time_arr=np.array(time_arr)
avg=np.average(time_arr[1:len(time_arr)])
fps=1/avg
print("average time of single one by one image inference is ")
print(str(avg), " seconds")
print("which translated to ",str(fps)," frames per second performance")
#input("continue")
print("")

interpreter.resize_tensor_input(input_index,[2,112,112,3])
interpreter.allocate_tensors()
print("")
time_arr=[]
print("testing sending 2 files at a time")
for i in range(5):
    temp=i_640[i:i+2]
    temp2=temp.astype(np.float32)
    start_time()
    interpreter.set_tensor(input_index, temp2)
    interpreter.invoke()
    pred_bird = interpreter.get_tensor(output_index)
    end_time()
    time_arr+=[diff]
time_arr=np.array(time_arr)
avg=np.average(time_arr)
fps=2/avg
print("average time of 2 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ",str(fps)," frames per second performance")
#input("continue")
print("")

interpreter.resize_tensor_input(input_index,[4,112,112,3])
interpreter.allocate_tensors()
print("")
time_arr=[]
print("testing sending 4 files at a time")
for i in range(5):
    temp=i_640[i:i+4]
    temp2=temp.astype(np.float32)
    start_time()
    interpreter.set_tensor(input_index, temp2)
    interpreter.invoke()
    pred_bird = interpreter.get_tensor(output_index)
    end_time()
    time_arr+=[diff]
time_arr=np.array(time_arr)
avg=np.average(time_arr)
fps=4/avg
print("average time of 4 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ",str(fps)," frames per second performance")
#input("continue")
print("")


interpreter.resize_tensor_input(input_index,[8,112,112,3])
interpreter.allocate_tensors()
print("")
time_arr=[]
print("testing sending 8 files at a time")
for i in range(5):
    temp=i_640[i:i+8]
    temp2=temp.astype(np.float32)
    start_time()
    interpreter.set_tensor(input_index, temp2)
    interpreter.invoke()
    pred_bird = interpreter.get_tensor(output_index)
    end_time()
    time_arr+=[diff]
time_arr=np.array(time_arr)
avg=np.average(time_arr)
fps=8/avg
print("average time of 8 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ",str(fps)," frames per second performance")
#input("continue")
print("")



interpreter.resize_tensor_input(input_index,[16,112,112,3])
interpreter.allocate_tensors()
print("")
time_arr=[]
print("testing sending 16 files at a time")
for i in range(5):
    temp=i_640[i:i+16]
    temp2=temp.astype(np.float32)
    start_time()
    interpreter.set_tensor(input_index, temp2)
    interpreter.invoke()
    pred_bird = interpreter.get_tensor(output_index)
    end_time()
    time_arr+=[diff]
time_arr=np.array(time_arr)
avg=np.average(time_arr)
fps=16/avg
print("average time of 16 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ",str(fps)," frames per second performance")
#input("continue")
print("")

interpreter.resize_tensor_input(input_index,[32,112,112,3])
interpreter.allocate_tensors()
print("")
time_arr=[]
print("testing sending 32 files at a time")
for i in range(5):
    temp=i_640[i:i+32]
    temp2=temp.astype(np.float32)
    start_time()
    interpreter.set_tensor(input_index, temp2)
    interpreter.invoke()
    pred_bird = interpreter.get_tensor(output_index)
    end_time()
    time_arr+=[diff]
time_arr=np.array(time_arr)
avg=np.average(time_arr)
fps=32/avg
print("average time of 32 images at a time inference is ")
print(str(avg), " seconds")
print("which translated to ",str(fps)," frames per second performance")
#input("continue")
print("")






