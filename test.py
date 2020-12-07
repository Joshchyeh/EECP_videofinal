from data import *
from keras.models import load_model
import pandas as pd

# data path
dat_path = 'train/RaceHorses_416x240_30.yuv'
lab_path = 'train/RaceHorses_832x480_30.yuv'
vdsr_path = 'recon/RaceHoreses_vdsr.yuv'
bicubic_path = 'recon/RaceHorses_bicubic.yuv'

# load model
model = load_model('my_model.h5')

# read data, preprocess, and predict
input_dat = read_yuv(dat_path, 416, 240, 300)
dat = gen_data(input_dat)
crop_dat = crop(dat, 832, 480, 300)[:, :, :, :, np.newaxis]
pre = np.zeros(crop_dat.shape)
for i in range(300):
    pre[i, :, :, :, :] = model.predict(crop_dat[i, :, :, :, :])
pre_dat = np.zeros((300, 480, 832))
for i in range(300):
    pre_dat[i, :, :] = crop_inv(pre[i, :, :, :, :], 832, 480)

pre_dat[pre_dat > 1.0] = 1.0
pre_dat[pre_dat < 0.0] = 0.0

# write reconstruct file and record psnr values
psnr = np.zeros((300, 2))
lab = np.zeros((480, 832))
f_lab = open(lab_path, 'rb')
f_out1 = open(vdsr_path, 'wb')
f_out2 = open(bicubic_path, 'wb')
for i in range(300):
    for j in range(480):
        for k in range(832):
            lab[j, k] = ord(f_lab.read(1)) / 255.0
            f_out1.write(chr(int(pre_dat[i, j, k] * 255)).encode('latin1'))
            f_out2.write(chr(int(dat[i, j, k] * 255)).encode('latin1'))
    uv = f_lab.read(int(0.5 * 832 * 480))
    f_out1.write(uv)
    f_out2.write(uv)
    psnr[i, 0] = -10 * np.log10(np.mean((dat[i, :, :] - lab) ** 2))
    psnr[i, 1] = -10 * np.log10(np.mean((pre_dat[i, :, :] - lab) ** 2))

f_lab.close()
f_out1.close()
f_out2.close()

# write PSNR value of every frame to excel file
psnr_df = pd.DataFrame(psnr)
psnr_df.columns = ['bicubic', 'vdsr']
psnr_df.to_excel('psnr.xlsx')
