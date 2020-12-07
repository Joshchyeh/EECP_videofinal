from keras.models import Model
from keras.layers import Conv2D, Input, Add
from keras.optimizers import Adam, SGD

# define VDSR model
def vdsr(input_size = (32, 32, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1')(inputs)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2')(conv1)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3')(conv2)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4')(conv3)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5')(conv4)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6')(conv5)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7')(conv6)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8')(conv7)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9')(conv8)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10')(conv9)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv11')(conv10)
    conv12 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv12')(conv11)
    conv13 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv13')(conv12)
    conv14 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv14')(conv13)
    conv15 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv15')(conv14)
    conv16 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv16')(conv15)
    conv17 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv17')(conv16)
    conv18 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv18')(conv17)
    conv19 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv19')(conv18)
    conv20 = Conv2D(1, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv20')(conv19)
    out = Add()([inputs, conv20])
    
    model = Model(input = inputs, output = out)
    model.compile(optimizer = Adam(lr = 5e-4), loss = 'mse')
    
    return model
