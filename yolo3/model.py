# coding: utf-8
from functools import reduce

from keras.layers import Input
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Model, save_model
from keras.regularizers import l2


def compose(*funcs):
    return reduce(lambda f, g: lambda a: g(f(a)), funcs)


def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    kws = {'kernel_regularizer': l2(5e-4)}
    kws['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    kws.update(kwargs)
    return Conv2D(*args, **kws)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    kws = {'use_bias': False}
    kws.update(kwargs)
    return compose(
        DarknetConv2D(*args, **kws),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    num_anchors = 3
    inputs = Input(shape=(None, None, 3))
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(
        darknet.output, 512, num_anchors * (4 + 1 + num_classes))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (4 + 1 + num_classes))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (4 + 1 + num_classes))

    return Model(inputs, [y1, y2, y3])


if __name__ == '__main__':
    save_path = './model_data/yolov3.h5'
    class_path = './model_data/classes.txt'

    classes = [c.strip() for c in open(class_path)]
    model = yolo_body(num_classes=len(classes))
    save_model(model, save_path)
