import cv2
import numpy as np
import pandas as pd
import random
from augment import augment_brightness as ab

train_batch_pointer = 0
val_batch_pointer = 0


train_df = pd.read_csv('../recvd_data/final.csv')
xs = train_df.id.values
ys = train_df.angle.values


num_images = len(xs)

c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs)*0.8)]
train_ys = ys[:int(len(xs)*0.8)]

val_xs = xs[-int(len(xs)*0.2):]
val_ys = ys[-int(len(xs)*0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


def LoadTrainBatch(batch_size):

    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img = cv2.resize(cv2.imread('../recvd_data/final_data/'+train_xs[(train_batch_pointer + i) % num_train_images]+'.jpg'), (320, 240))
        ang = [train_ys[(train_batch_pointer + i) % num_train_images]]

        img = ab(img)
        if np.random.randint(2):
            img = np.fliplr(img)
            ang[0] = -ang[0]

        x_out.append(img/ 255.)
        y_out.append(ang)
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):

    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img = cv2.resize(cv2.imread('../recvd_data/final_data/'+val_xs[(val_batch_pointer + i) % num_val_images]+'.jpg'), (320, 240))
        ang = [val_ys[(val_batch_pointer + i) % num_val_images]]

        img = ab(img)
        if np.random.randint(2):
            img = np.fliplr(img)
            ang[0] = -ang[0]

        x_out.append(img/ 255.)
        y_out.append(ang)
    val_batch_pointer += batch_size
    return x_out, y_out
