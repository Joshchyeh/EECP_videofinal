# EECP_videofinal
Topic: VDSR - application on video\
Team: 10\
Member: B04901189 周忠履、B04204037 葉承鑫
# Data
* Download data from here: <https://drive.google.com/drive/folders/1J7JO2p8xaCnc8ATi1VwHUdsftIy1V2w2>
  * You will see two directories(train and recon) after you click the link, please download both of them
    * In train directory, there are two .yuv files with different resolution (416x240 and 832x480), those are our training data and ground truth
    * In recon directory, there are two .yuv files with same resolution (832x480), those are our result by using bicubic interpolation and VDSR
  * After download, please put both directories in the same directory (train and recon) with all .py files
# Execution
* Required modules:
  * tensorflow
  * keras
  * numpy
  * pandas
  * matplotlib
  * skimage
  * os
* Training: 
  * Execute main.py
  * Some remind:
    1. If you want to save your own model, remember to uncomment "model.save('my_model.h5')"
    2. If you want to save the training curve picture, remember to uncomment "plt.savefig('vdsr_loss.png')"
    3. If you want to check the training curve picture during execution, remember uncomment "plt.show()" (this will block the execution)
    4. If you want to use gpu, please check main.py, there are some codes provided for gpu setting, just uncomment them, and set the parameters
* Testing:
  * Execute test.py
  * You will got two .yuv files (in recon directory) and a .xlsx file after execution
  * Some remind:
    1. If you do not want to output the excel file which stores the PSNR value of both bicubic and VDSR methods of each frame, please comment out the last 3 lines in test.py
* Both executions will take several time, please be patient or try to run on better devices (I run on Intel Xeon CPU E5-2630 v2 + GTX980)
* There might have some warnings during execution, but those warning will not affect the result, so just ignore them
# Visualization
* Because all videos in my project are .yuv files, so if you want to visualize the result, you can:
   1. Download yuv player: <https://sourceforge.net/projects/raw-yuvplayer/files/latest/download>
   2. Convert all yuv files into other forms of video files by yourself
* Remind: When playing the .yuv files, remember to adjust the player's frame size to correct size (in this case, only RaceHorses_416x240_30.yuv has frame size 416x240, others' frame size are all 832x480)
# Some results
1. Loss curve:\
![](https://github.com/jz0831/eecp_final/blob/master/vdsr_loss.png)
2. Comparison betwwen frame number 125 of different videos (gif, priority: bicubic -> VDSR -> original)
  * Please look carefully to check the difference between results of different approach\
  ![](https://github.com/jz0831/eecp_final/blob/master/2lfdw-23nmj.gif)
  * My observations:
    1. Obviously, the resolution of VDSR result is better than bicubic result
    2. Most obvious difference between VDSR result and original data is that the background(grass) in VDSR result is blur
    3. Objects in bicubic result actually shift a little bit while VDSR result does not (compare to original data)
