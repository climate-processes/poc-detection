import sys

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Activation, Concatenate, Add, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

sys.setrecursionlimit(3000)

# define the model for refining the mask
K.set_image_data_format('channels_last')
K.set_floatx('float32')
smallest_layer = 32


def RESUNETMEDIUM(input_shape, n_classes):
    inputs = Input(input_shape)

    ###encoding block 1
    iden1 = Conv2D(smallest_layer, 1, activation=None, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(smallest_layer, 3, activation=None, padding='same', kernel_initializer='he_normal')(inputs)  # 200
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(smallest_layer, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv1)  # 200
    add1 = Add()([iden1, conv1])
    pool1 = MaxPooling2D((3, 3))(add1)  # 224 -> 74
    print(add1)

    ###encoding block2
    iden2 = Conv2D(smallest_layer * 2, 1, activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(smallest_layer * 2, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)  # 200
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(smallest_layer * 2, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)  # 200
    add2 = Add()([iden2, conv2])
    pool2 = MaxPooling2D((3, 3))(add2)  # 74 -> 24
    print(add2)

    ###encoding block4
    iden4 = Conv2D(smallest_layer * 8, 1, activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv4 = BatchNormalization()(pool2)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(smallest_layer * 8, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)  # 200
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(smallest_layer * 8, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)  # 200
    add4 = Add()([iden4, conv4])
    drop4 = Dropout(0.5)(add4)
    pool4 = MaxPooling2D((3, 3))(drop4)  # 24 -> 8
    print(pool4)

    ###bridge
    conv5 = BatchNormalization()(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(smallest_layer * 16, 3, activation=None, padding='same', kernel_initializer='he_normal')(
        conv5)  # 200
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(smallest_layer * 16, 3, activation=None, padding='same', kernel_initializer='he_normal')(
        conv5)  # 200
    drop5 = Dropout(0.5)(conv5)
    print(conv5)

    ###decoding block1
    up6 = UpSampling2D((3, 3))(drop5)  # 8->24
    # up6 = ZeroPadding2D(((1,0),(1,0)))(up6) #24->25
    concat6 = Concatenate(axis=3)([up6, add4])
    iden6 = Conv2D(smallest_layer * 8, 1, activation=None, padding='same', kernel_initializer='he_normal')(concat6)
    conv6 = BatchNormalization()(concat6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(smallest_layer * 8, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv6)  # 200
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(smallest_layer * 8, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv6)  # 200
    add6 = Add()([iden6, conv6])

    ###decoding block3
    up8 = UpSampling2D((3, 3))(add6)  # 24 -> 72
    up8 = ZeroPadding2D(((0, 2), (0, 2)))(up8)  # 72->74
    concat8 = Concatenate(axis=3)([up8, add2])
    iden8 = Conv2D(smallest_layer * 2, 1, activation=None, padding='same', kernel_initializer='he_normal')(concat8)
    conv8 = BatchNormalization()(concat8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(smallest_layer * 2, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv8)  # 200
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(smallest_layer * 2, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv8)  # 200
    add8 = Add()([iden8, conv8])

    ###decoding block4
    up9 = UpSampling2D((3, 3))(add8)  # 74 -> 222
    up9 = ZeroPadding2D(((0, 2), (0, 2)))(up9)  # 222->224
    concat9 = Concatenate(axis=3)([up9, add1])
    iden9 = Conv2D(smallest_layer, 1, activation=None, padding='same', kernel_initializer='he_normal')(concat9)
    conv9 = BatchNormalization()(concat9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(smallest_layer, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv9)  # 200
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(smallest_layer, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv9)  # 200
    add9 = Add()([iden9, conv9])

    conv10 = Conv2D(n_classes, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(add9)
    # sigmoid probably too strong an activation

    model = Model(inputs, conv10)

    return model

# define the dice coefficient used
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def get_model(weights_file=None, learning_rate=0.001, weight_decay=0.0):
    # create the refined mask model and load the weights

    refined_mask_model = RESUNETMEDIUM((224, 224, 3), 1)

    if weights_file is not None:
        refined_mask_model.load_weights(weights_file)

    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=weight_decay, amsgrad=True)
    refined_mask_model.compile(optimizer=optimizer,
                               loss=dice_coef_loss, metrics=[dice_coef, 'accuracy', 'binary_crossentropy'])

    return refined_mask_model
