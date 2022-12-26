from tensorflow.keras import Model

#Calculate the cost
def cost_compute(s3ar_model):
    from keras_flops import get_flops
    from keras.utils.layer_utils import count_params
    flops  = float("{0:.2f}".format(get_flops(Model(s3ar_model.input, s3ar_model.output), batch_size=1)/ 10 ** 9))
    print("FLOPS:", flops, "GFLOPS")

#Prepare loss functions and metrics
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

