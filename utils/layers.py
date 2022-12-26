from utils import parameters
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import *

#Kernel size
ks = 5

#SE Ratio
ratio = 8

#Filter multiplier (fm)
fm = 1

#DSConv filter size
ds_f = 8

ds_f = ds_f * fm

#Kernel Initializer
ki = 'he_normal'

#Activation
activation = 'relu6'

#Squeeze-and-excitation blocks
def squeeze_excite_block(inputs, ratio=ratio):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation=activation, kernel_initializer=ki, use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=ki, use_bias=False)(se)
    
    x = Multiply()([init, se])
    return x

def spatial_squeeze_excite_block(inputs):
    se = SeparableConv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer=ki)(inputs)
    
    x = multiply([inputs, se])
    return x

def channel_spatial_squeeze_excite(inputs, ratio=ratio):
    cse = squeeze_excite_block(inputs, ratio=ratio)
    sse = spatial_squeeze_excite_block(inputs)
    
    x = add([cse, sse])
    return x


#Stem block
def stem_block(x, n_filter, strides):
    x_init = x
    
    ## Conv 1
    x = SeparableConv2D(n_filter, (ks, ks), kernel_initializer=ki, padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SeparableConv2D(n_filter, (ks, ks),kernel_initializer=ki, padding="same")(x)
    
    ## Shortcut
    s = SeparableConv2D(n_filter, (1, 1),kernel_initializer=ki, padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)
    
    ## Add
    x = Add()([x, s])
    x = channel_spatial_squeeze_excite(x, ratio=ratio)
    return x


#ResBlock
def resnet_block(x, n_filter, strides=1):
    x_init = x
    
    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SeparableConv2D(n_filter, (ks, ks), kernel_initializer=ki, padding="same", strides=strides)(x)
    
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = SeparableConv2D(n_filter, (ks, ks), kernel_initializer=ki, padding="same", strides=1)(x)
    
    ## Shortcut
    s = SeparableConv2D(n_filter, (1, 1), kernel_initializer=ki, padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)
    
    ## Add
    x = Add()([x, s])
    x = channel_spatial_squeeze_excite(x, ratio=ratio)
    return x


#Attention
def attention_block(g, x):

    filters = x.shape[-1]
    
    g_conv = BatchNormalization()(g)
    g_conv = Activation(activation)(g_conv)
    g_conv = SeparableConv2D(filters, (ks, ks), kernel_initializer=ki, padding="same")(g)
    
    g_pool = DepthwiseConv2D(filters, (4, 4), kernel_initializer=ki, padding="same")(g_conv)
    g_pool = UpSampling2D((2, 2))(g_pool)
    
    x_conv = BatchNormalization()(x)
    x_conv = Activation(activation)(x_conv)
    x_conv = SeparableConv2D(filters, (ks, ks), kernel_initializer=ki, padding="same")(x)
    
    gc_sum = Add()([g_pool, x_conv])
    
    gc_conv = Activation(activation)(gc_sum)
    gc_conv = SeparableConv2D(filters, (1, 1), kernel_initializer=ki, padding="same")(gc_conv)
    gc_conv = Activation('sigmoid')(gc_conv)
    
    gc_mul = Multiply()([gc_conv, x])
    gc_mul = channel_spatial_squeeze_excite(gc_mul, ratio=ratio)
    return gc_mul


def single_conv_block(inputs, filters):
    x = SeparableConv2D(filters, (ks, ks), padding="same", activation=activation)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def fsm(inputs):
    channel_num = inputs.shape[-1]

    res = inputs

    inputs = single_conv_block(inputs, filters=int(channel_num // 2))

    ip = inputs
    ip_shape = K.int_shape(ip)
    batchsize, dim1, dim2, channels = ip_shape
    intermediate_dim = channels // 2
    rank = 4
    if intermediate_dim < 1:
        intermediate_dim = 1
    
    # theta path
    theta = SeparableConv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer=ki,
                   kernel_regularizer=l2(1e-5))(ip)
    theta = Reshape((-1, intermediate_dim))(theta)
    
    # phi path
    phi = SeparableConv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer=ki,
                   kernel_regularizer=l2(1e-5))(ip)
    phi = Reshape((-1, intermediate_dim))(phi)
    
    # dot
    f = dot([theta, phi], axes=2)
    size = K.int_shape(f)
    # scale the values to make it size invariant
    f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)
    
    # g path
    g = SeparableConv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer=ki,
                   kernel_regularizer=l2(1e-5))(ip)
    g = Reshape((-1, intermediate_dim))(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])
    y = Reshape((dim1, dim2, intermediate_dim))(y)
    y = SeparableConv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer=ki,
               kernel_regularizer=l2(1e-5))(y)
    y = add([ip, y])

    x = y
    x = single_conv_block(x, filters=int(channel_num))
    print(x)

    x = add([x, res])
    return x
