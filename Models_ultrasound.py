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

# set seed
tf.random.set_seed(1203)

as_gray = True
in_channel = 3
img_rows, img_cols = 256, 256
num_classes = 4   # 2 for "Luminal vs Non-Luminal"
batch_size = 32
all_epochs = 200
input_shape = (img_rows, img_cols, in_channel)
input_img = Input(shape = input_shape)

# Loading data
train_df = pd.read_csv('.../train_labels.csv', index_col=0)
train_df['US_file'] = train_df.index.map(lambda id: f'.../Multimodal_data/train/{id}_US.png')

x_data=train_df.iloc[:,1:2]
y_data=train_df.iloc[:,0:1] 

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.2, random_state=1203)
print("Uploading train_us...")
x_train_US = read_us(x_train.US_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
y_train = y_train.appliance.values
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)

print("Uploading val_us...")
x_val_US = read_us(x_val.US_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")
y_val = y_val.appliance.values
y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes)
print("------------------------------------------------------------------------------------------------")

# f1_score
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

# Learning_rate_metric
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

  
# Model

base1=ResNet50(weights='imagenet',include_top=False,input_shape=(256, 256, 3))

for layer in base1.layers :
    layer._name = layer.name + str('_1')

US_model = Model(inputs=base1.input, outputs=base1.get_layer('conv4_block6_out_1').output)

x=US_model.output

x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

x = GlobalAveragePooling2D()(x)
x = Flatten()(x) 
x = Dense(512,"relu")(x)   
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax',name='output')(x)
model = Model(inputs=[US_model.input], outputs=[output])
opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
lr_metric = get_lr_metric(opt)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[f1, 'accuracy', lr_metric])   # loss='binary_crossentropy' for "Luminal vs Non-Luminal"

def scheduler(epoch):
    if (epoch) % (10) == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.9)
        print("lr changed to {}".format(lr * 0.9))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

# path to save the model
best_weights_file=".../US.weights.best.hdf5"

# checkpoint, callbacks 
checkpoint = ModelCheckpoint(best_weights_file, monitor='val_f1', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks = [checkpoint, reduce_lr]

# Training model
history=model.fit(x_train_US,y_train,batch_size=batch_size, epochs=all_epochs, callbacks=callbacks, verbose=1, validation_data=(x_val_US,y_val),shuffle=True)

# Test model
test_df = pd.read_csv('.../test_labels.csv', index_col=0)
test_df['US_file'] = test_df.index.map(lambda id: f'.../Multimodal_data/test/{id}_US.png')
x_test=test_df.iloc[:,1:2]
y_test=test_df.iloc[:,0:1]
print("Uploading test_us...")
x_test_US = r_us(x_test.US_file.values, img_rows, img_cols, as_gray, in_channel)
print("Done!")
print("------------------------------------------------------------------------------------------------")

y_test = y_test.appliance.values
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

model.load_weights(best_weights_file)

predictions = model.predict([x_test_US],batch_size=32)
y_test
y_test=y_test.argmax(axis=1)
y_pred = np.rint(predictions)
y_pred=y_pred.argmax(axis=1)
matrix=metrics.confusion_matrix(y_test,y_pred)
print("matrix=:")
print(matrix)
#MCC=matthews_corrcoef(y_test, y_pred)
