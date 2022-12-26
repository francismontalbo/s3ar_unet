from utils.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras import Model

#Image
image_rows = int(256/2)
image_cols = int(256/2) 
IMAGE_SIZE = image_rows
image_depth = 3
fm = 1

def s3ar_unet():
    n_filters = [8 * fm, 16 * fm, 32 * fm, 64 * fm]
    inputs = Input((image_rows, image_cols, 3))
    
    ## Input Layer
    input_layer = inputs 
    
    ## Stem
    stem = stem_block(input_layer, n_filters[3], strides=2)
    stem = DepthwiseConv2D(2, 2, padding='same')(stem)
    
    ## Encoder
    e1 = resnet_block(stem, n_filters[2], strides=1)
    e1 = fsm(e1)
    e1 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(e1)
    e1 = MaxPooling2D(pool_size=(2, 2), strides=2)(e1)
    
    e2 = resnet_block(e1, n_filters[2], strides=1)
    e2 = fsm(e2)
    e2 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(e2)
    e2 = MaxPooling2D(pool_size=(2, 2), strides=2)(e2)
    
    e3 = resnet_block(e2, n_filters[1], strides=1)
    e3 = fsm(e3)
    e3 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(e3)
    e3 = MaxPooling2D(pool_size=(2, 2), strides=2)(e3)
    
    e4 = resnet_block(e3, n_filters[0], strides=1)
    e4 = fsm(e4)
    e4 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(e4)
    e4 = MaxPooling2D(pool_size=(2, 2), strides=2)(e4)
    
    ## Brdige
    b1 = fsm(e4)
    b1 = SeparableConv2D(n_filters[0], kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(b1)
    b1 = channel_spatial_squeeze_excite(b1, ratio=ratio)
    
    ## Decoder
    d1 = attention_block(e3, b1)
    d1 = Conv2DTranspose(n_filters[0], (2, 2), strides=2)(d1)
    d1 = Add()([d1, e3])
    d1 = resnet_block(d1, n_filters[0])
    d1 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(d1)
    d1 = channel_spatial_squeeze_excite(d1, ratio=ratio)
    
    d2 = attention_block(e2, d1)
    d2 = Conv2DTranspose(n_filters[0], (2, 2), strides=2)(d2)
    d2 = Add()([d2, e2])
    d2 = resnet_block(d2, n_filters[1])
    d2 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(d2)
    d2 = channel_spatial_squeeze_excite(d2, ratio=ratio)
    
    d3 = attention_block(e1, d2)
    d3 = Conv2DTranspose(n_filters[0], (2, 2), strides=2)(d3)
    d3 = Add()([d3, e1])
    d3 = resnet_block(d3, n_filters[2])
    d3 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(d3)
    d3 = channel_spatial_squeeze_excite(d3, ratio=ratio)
    
    d4 = attention_block(stem, d3)
    d4 = Conv2DTranspose(n_filters[0], (2, 2), strides=2)(d4)
    d4 = Concatenate()([d4, stem])
    d4 = resnet_block(d4, n_filters[2])
    d4 = SeparableConv2D(ds_f, kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(d4)
    d4 = channel_spatial_squeeze_excite(d4, ratio=ratio)
    
    #Output Block
    outputs = SeparableConv2D(n_filters[3], kernel_size=(ks, ks), kernel_initializer=ki, padding='same')(d4)
    outputs = UpSampling2D((4, 4))(outputs)
    outputs = Concatenate()([outputs, input_layer])
    outputs = channel_spatial_squeeze_excite(outputs, ratio=ratio)
    outputs = SeparableConv2D(1, (ks, ks), padding="same")(outputs)
    outputs = Activation("sigmoid")(outputs)
    
    ## Model
    model = Model(inputs, outputs, name="S3AR_UNet")
    return model