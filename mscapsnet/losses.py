from keras import backend as K


def margin_loss(y_true, y_pred):
    L = (y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 *
         (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1)))
    return K.mean(K.sum(L, 1))
