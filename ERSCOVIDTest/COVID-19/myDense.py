"""
"""
        
from __future__ import print_function
from __future__ import absolute_import

#"""
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import AveragePooling3D, GlobalAveragePooling3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.utils import plot_model
#"""

from tensorflow.keras import backend as K

def _conv_block(x, filterNum, name):
    """
    """
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv3D(filterNum, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same', name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(filterNum, (3, 3, 3), strides=(1, 1, 1), padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def _transit_block(x, filterNum, name):
    """
    """
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_trans_bn')(x)
    x = Activation('relu', name=name + '_trans_relu')(x)
    x = Conv3D(filterNum, (1, 1, 1), strides=(1, 1, 1), use_bias=False, padding='same', name=name + '_trans_conv')(x)
    return x

"""
blocks is the filterNum list in a block, e.g., [32, 32]
name = 'block_0'
"""
def _denseBlock(x, blocks, name):
    for ie, i in enumerate(blocks):
        x = _conv_block(x, i, name=name+'_'+str(ie))
    return x

def myDenseNetv2(input_shape):
    """
    """
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1

    input_tensor = Input(shape=input_shape)#48, 240, 360, 1
    x = Conv3D(16, (3, 3, 3), strides=(1, 2, 2), use_bias=False, padding='same', name='block0_conv1')(input_tensor)#[48, 120, 180]
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='block0_bn1')(x)
    x = Activation('relu', name='block0_relu1')(x)
    x = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), use_bias=False, padding='same', name='block0_conv2')(x)#[48, 120, 180]
    
    x = _denseBlock(x, [16, 16], 'block_11')#[48, 120, 180]
    x = _transit_block(x, 16, 'block13')#[48, 120, 180]
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)#[24, 60, 90]
    
    x = _denseBlock(x, [24, 24, 24], 'block_21')#[24, 60, 90]
    x =_transit_block(x, 24, 'block23')#[24, 60, 90]
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)#[12, 30, 45]
    
    x = _denseBlock(x, [32, 32, 32, 32], 'block_31')#[12, 30, 45]
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)#[6, 15, 23]
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='block_final_bn')(x)
    x = Activation('relu', name='block_final_relu')(x)
    
    ##############above are bae####################
    x = _denseBlock(x, [32, 32], 'EGFR_block_11')
    x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), padding='same')(x)#[6, 8, 12, 64]
    x = _transit_block(x, 64, 'EGFR_block_12')#[6, 8, 12, 64]
    x = GlobalAveragePooling3D()(x)
    x = Dense(1, activation='sigmoid', name='EGFR_global_pred')(x)

    # create model
    model = Model(input_tensor, x, name='myDense')
    #plot_model(model, 'myDenseNetv2.png', show_shapes=True)

    return model

if __name__ == '__main__':
    myDenseNetv2((48, 240, 360, 1))
    