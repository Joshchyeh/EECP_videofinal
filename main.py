import matplotlib.pyplot as plt
# for gpu user, you can decide which gpu you want to use, and the memory fraction
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#sess = tf.Session(config = config)
#KTF.set_session(sess)
from data import *
from vdsr import *

# read training data and label
dat_path = 'train/RaceHorses_416x240_30.yuv'
lab_path = 'train/RaceHorses_832x480_30.yuv'
y_dat = read_yuv(dat_path, 416, 240, 300)
y_lab = read_yuv(lab_path, 832, 480, 300)
#print(y_dat.shape)
#print(y_lab.shape)

# apply bicubic interpolation on low resolution image
y_inter = gen_data(y_dat)
#print(y_inter.shape)

# crop both data and label
y_crop_dat = crop(y_inter, 832, 480, 300)
y_crop_lab = crop(y_lab, 832, 480, 300)
#print(y_crop_dat.shape, y_crop_lab.shape)

# split data for training and validation and testing
train_dat = y_crop_dat[0, :, :][:, :, :, np.newaxis]
train_lab = y_crop_lab[0, :, :][:, :, :, np.newaxis]
val_dat = y_crop_dat[1, :, :][:, :, :, np.newaxis]
val_lab = y_crop_lab[1, :, :][:, :, :, np.newaxis]
test_dat = y_crop_dat[2::, :, :].reshape((298 * 390, 32, 32, 1))
test_lab = y_crop_lab[2::, :, :].reshape((298 * 390, 32, 32, 1))
#print(train_dat.shape, train_lab.shape)
#print(val_dat.shape, val_lab.shape)
#print(test_dat.shape, val_lab.shape)

# start training
model = vdsr()
epoch = 50
history = model.fit(train_dat, train_lab, epochs = epoch, verbose = 1, validation_data = (val_dat, val_lab))

# plot loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
# plt.savefig('vdsr_loss.png')
# plt.show()

# save model
#model.save('my_model.h5')
