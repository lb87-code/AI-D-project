import tensorflow as tf
import func_lib as fl
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy
import numpy as np
import logging
import pandas as pd

tf.get_logger().setLevel(logging.ERROR)
#資料範圍
f_range = {'f_min':20e3,'f_max':200e3}
c_range = {'c_min':47e-6,'c_max':726e-6}
l_range = {'l_min':100e-6,'l_max':1000e-6}

# 載入訓練資料集(inductor loss)
data_train_indl = pd.read_csv(
    r".\train_data\final_data\indl.csv",
    names=["fsw", "cap", "ind", "pl_i_cu"])

# 訓練資料集格式
data_features_indl = data_train_indl.copy()
data_labels_indl = data_features_indl.pop('pl_i_cu')
data_features_indl = np.array(data_features_indl)

#輸入資料歸一化
data_features_indl['fsw'] = (data_features_indl['fsw']-f_range['f_min'])/(f_range['f_max']-f_range['f_min'])
data_features_indl['cap'] = (data_features_indl['cap']-c_range['c_min'])/(c_range['c_max']-c_range['c_min'])
data_features_indl['ind'] = (data_features_indl['ind']-l_range['l_min'])/(l_range['l_max']-l_range['l_min'])
data_features=data_features_indl

#inductor loss BN-NN
initializer1 = keras.initializers.he_normal()

initializer2 = keras.initializers.he_normal()

initializer3 = keras.initializers.he_normal()

initializer4 = keras.initializers.he_normal()

bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

ACTIVATIONS = 'relu'
ACTIVATIONS_OUT = 'relu'

model_indl_cu = keras.Sequential([

   #BN-NN layer 1 (20個神經元)
   keras.layers.Dense(20, activation='softmax',
                       kernel_initializer=initializer1,
                       bias_initializer=bias_initializer),
   keras.layers.BatchNormalization(),
   #BN-NN layer 2 (20個神經元)
   keras.layers.Dense(20, activation='softmax',
                       kernel_initializer=initializer2,
                       bias_initializer=bias_initializer),
   keras.layers.BatchNormalization(),
   #BN-NN layer 3 (20個神經元)
   keras.layers.Dense(20, activation='softmax',
                       kernel_initializer=initializer3,
                       bias_initializer=bias_initializer),
   keras.layers.BatchNormalization(),
   #output layer (直接輸出)
   keras.layers.Dense(1, activation='relu',
                       kernel_initializer=initializer4,
                       bias_initializer=bias_initializer)
])

EPOCHS = 10 #設定訓練總週期
BATCH_SIZE = 64 #每小批次取1個樣本來訓練
LOSS = 'mean_squared_error' #損失函數
LOSS2 = 'binary_crossentropy'
opt4 = keras.optimizers.SGD(learning_rate=0.01)
model_indl_cu.compile(loss=LOSS, optimizer = opt4)

history_indl_cu = model_indl_cu.fit(data_features, data_labels_indl,
                    validation_split = 0.3,
                    epochs=15, batch_size=BATCH_SIZE
                   )

predict = model_indl_cu.predict(data_features)
r2 = r2_score(data_labels_indl, predict)

print(f"R²: {r2:.2f}")



fl.print_history(history_indl_cu,'inductor copper loss',1,0.5)
