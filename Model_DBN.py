import numpy as np
from DBN_Models import SupervisedDBNClassification
from Evaluation import evaluation


def Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target):
    classifier = SupervisedDBNClassification(hidden_layers_structure=[128, 128],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.01,
                                             n_epochs_rbm=1,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)

    Y_pred = np.zeros(Test_Target.shape)
    for i in range(Test_Target.shape[1]):
        classifier.fit(Train_Data, Train_Target[:, i])

        Y_pred[:, i] = classifier.predict(Test_Data)
    Y_pred = np.asarray(Y_pred)
    Y_pred[Y_pred < 0.5] = 0
    Y_pred[Y_pred >= 0.5] = 1
    Eval = evaluation(Y_pred, Test_Target)
    return Eval, Y_pred

