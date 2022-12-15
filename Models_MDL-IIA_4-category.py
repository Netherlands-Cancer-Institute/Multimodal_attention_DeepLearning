import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, initializers, regularizers, constraints, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation 
from tensorflow.keras.layers import concatenate, AveragePooling2D, ZeroPadding2D, add, Reshape
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib

from matplotlib import pyplot as plt
from sklearn import metrics, manifold
from sklearn.metrics import matthews_corrcoef  
from sklearn.model_selection import train_test_split
from Data_loading import read_mg, read_us
from Attention_layers import Self_Attention

### set seed
tf.random.set_seed(1203)

as_gray = True
in_channel = 3
img_rows, img_cols = 256, 256
num_classes = 4
batch_size = 32
all_epochs = 200
input_shape = (img_rows, img_cols, in_channel)
input_img = Input(shape = input_shape)

### Loading data
train_df = pd.read_csv('.../train_labels.csv', index_col=0)
train_df['MLO_file'] = train_df.index.map(lambda id: f'.../Multimodal_data/train/{id}_MLO.png')
train_df['CC_file'] = train_df.index.map(lambda id: f'.../Multimodal_data/train/{id}_CC.png')
train_df['US_file'] = train_df.index.map(lambda id: f'.../Multimodal_data/train/{id}_US.png')

x_data=train_df.iloc[:,1:4]
y_data=train_df.iloc[:,0:1] 

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.2, random_state=1203)
print("Uploading train_cc...")
x_train_CC = read_mg(x_train.CC_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
print("Uploading train_mlo...")
x_train_MLO = read_mg(x_train.MLO_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
print("Uploading train_us...")
x_train_US = read_us(x_train.US_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
y_train = y_train.appliance.values
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)

print("Uploading val_cc...")
x_val_CC = read_mg(x_val.CC_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
print("Uploading val_mlo...")
x_val_MLO = read_mg(x_val.MLO_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
print("Uploading val_us...")
x_val_US = read_us(x_val.US_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
y_val = y_val.appliance.values
y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes)
print("------------------------------------------------------------------------------------------------")

### f1_score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

###Learning_rate_metric
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

### channel_spatial_attention
channel_axis = 1 if K.image_data_format() == "channels_first" else 3
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)

    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

def CSA(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])

  
### Model
base1=ResNet50(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
base2=ResNet50(weights='imagenet',include_top=False,input_shape=(256, 256, 3))
base3=ResNet50(weights='imagenet',include_top=False,input_shape=(256, 256, 3))

for layer in base1.layers :
    layer._name = layer.name + str('_1')
for layer in base2.layers :
    layer._name = layer.name + str('_2')
for layer in base3.layers :
    layer._name = layer.name + str('_3')

MLO_model = Model(inputs=base1.input, outputs=base1.get_layer('conv4_block6_out_1').output)
CC_model = Model(inputs=base2.input, outputs=base2.get_layer('conv4_block6_out_2').output)
US_model = Model(inputs=base3.input, outputs=base3.get_layer('conv4_block6_out_3').output)
x_mlo=MLO_model.output
x_cc=CC_model.output
x_us=US_model.output
c1 = concatenate([x_mlo, x_cc],axis=2)

## intra-modality attention
a1=Self_Attention(1024)(c1)

x1 = tf.keras.layers.Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(a1)
x11,x12=x1[0], x1[1]
h1, w1, c1 = x11.shape[1],x11.shape[2],x11.shape[3]
x11=Reshape((h1, w1, c1))(x11)
x12=Reshape((h1, w1, c1))(x12)

x11 = bottleneck_Block(x11, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
x11 = bottleneck_Block(x11, nb_filters=[512, 512, 2048])
x11 = bottleneck_Block(x11, nb_filters=[512, 512, 2048])

x12 = bottleneck_Block(x12, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
x12 = bottleneck_Block(x12, nb_filters=[512, 512, 2048])
x12 = bottleneck_Block(x12, nb_filters=[512, 512, 2048])

# intra-modality attention
a2=Self_Attention(1024)(x_us)

x2 = bottleneck_Block(a2, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
x2 = bottleneck_Block(x2, nb_filters=[512, 512, 2048])
x2 = bottleneck_Block(x2, nb_filters=[512, 512, 2048])
ccc = concatenate([x11, x12, x2], axis=2)

# inter-modality attention
a3=Self_Attention(2048)(ccc)

x6 = tf.keras.layers.Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 3})(a3)
x61, x62, x63=x6[0], x6[1] ,x6[2]
h6, w6, c6 = x61.shape[1],x61.shape[2],x61.shape[3]
x61=Reshape((h6, w6, c6))(x61)
x62=Reshape((h6, w6, c6))(x62)
x63=Reshape((h6, w6, c6))(x63)
c6 = concatenate([x61, x62, x63], axis=3)

# channel and spatial attention
x3=CSA(c6)

x3 = tf.keras.layers.Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(x3)
x31, x32, x33=x3[0], x3[1] ,x3[2]
h3, w3, c3 = x31.shape[1],x31.shape[2],x31.shape[3]
x31=Reshape((h3, w3, c3))(x31)
x32=Reshape((h3, w3, c3))(x32)
x33=Reshape((h3, w3, c3))(x33)
x31 = GlobalAveragePooling2D()(x31)
x32 = GlobalAveragePooling2D()(x32)
x33 = GlobalAveragePooling2D()(x33)
x = concatenate([x31, x32, x33])
x = Flatten()(x) 
x = Dense(512,"relu")(x)   
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax',name='output')(x)
model = Model(inputs=[MLO_model.input,CC_model.input,US_model.input], outputs=[output])
opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
lr_metric = get_lr_metric(opt)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[f1, 'accuracy', lr_metric])

def scheduler(epoch):
    if (j) % (10) == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.9)
        print("lr changed to {}".format(lr * 0.9))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

### path to save the model
best_weights_file=".../MDL-IIA.weights.best.hdf5"

### checkpoint, callbacks 
checkpoint = ModelCheckpoint(best_weights_file, monitor='val_f1', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks = [checkpoint, reduce_lr]

### Training model
history=model.fit([x_train_MLO,x_train_CC,x_train_US],y_val,batch_size=batch_size, epochs=all_epochs, callbacks=callbacks, verbose=1, validation_data=([x_val_MLO,x_val_CC,x_val_US],y_val),shuffle=True)
