import numpy as np
from keras.layers import Dense, Bidirectional
from keras.models import Sequential
from keras.layers import Embedding, Dropout
from Evaluation import evaluation

def Model_BILSTM(train_data, train_target, test_data, test_target):
    train_data1 = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
    test_data1 = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
    model = Sequential()
    model.add(Embedding(train_data1.shape[0],1,input_length=(1, train_data1.shape[2])))
    model.add(Bidirectional(10))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'])
    model.fit(train_data1, train_target, epochs=100, batch_size=64, verbose=2)
    predict = model.predict(test_data1)
    Eval = evaluation(predict, test_target)
    return np.asarray(Eval).ravel(), predict

