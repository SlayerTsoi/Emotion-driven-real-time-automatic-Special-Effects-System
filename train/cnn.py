# models/cnn.py
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.layers import Activation, GlobalAveragePooling2D, Dense, Add, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def enhanced_XCEPTION(input_shape=(48,48,3), num_classes=7, l2_reg=1e-4):
    inputs = Input(shape=input_shape)
    
    # 初始块
    x = Conv2D(32, (3,3), strides=2, padding='same', 
              kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 中间块
    for filters in [64, 128, 256]:
        residual = Conv2D(filters, (1,1), strides=2, padding='same')(x)
        
        x = SeparableConv2D(filters, (3,3), padding='same',
                          depthwise_regularizer=l2(l2_reg),
                          pointwise_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, (3,3), padding='same',
                          depthwise_regularizer=l2(l2_reg),
                          pointwise_regularizer=l2(l2_reg))(x)
        x = MaxPooling2D((3,3), strides=2, padding='same')(x)
        x = Add()([x, residual])
    
    # 输出块
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='Enhanced_Xception')