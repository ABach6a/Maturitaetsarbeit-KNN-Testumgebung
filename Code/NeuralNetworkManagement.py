import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
import numpy as np

keras.backend.set_floatx('float64')

class NNManager: 

    def __init__(self): 
        print("NNManager initializing")
        self.current_run_topology = None
        self.current_run_learning_history = None
        self.current_run_holdout_accuracy = None
        self.current_run_roc_data = None

    def createNetworkModel(self, hiddenLayersConfig : list, withNonLinearActivation = True, withDropOutLayers = True): 
        print(f"creating network with hidden layers topology of sizes {hiddenLayersConfig}")
        self.current_run_topology = hiddenLayersConfig

        # remove existing network from memory
        keras.backend.clear_session()

        # define the keras model
        self.nn_model = keras.Sequential()
        if not withNonLinearActivation:
            self.addLayers(self.nn_model, hiddenLayersConfig, 'linear', withDropOutLayers)
            print("configured with linear !")
        else:
            self.addLayers(self.nn_model, hiddenLayersConfig, 'relu', withDropOutLayers)

        # compile the keras model
        self.nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # print a topology summary to the console
        self.nn_model.summary()


    def addLayers(self, model, hiddenLayersConfig, hiddenLayersActivationFunction, withDropOutLayers):
        # input layer
        model.add(layers.InputLayer(input_shape=(8,), name='input-layer'))
        # add hidden layers
        for i in range(0, len(hiddenLayersConfig)):
            model.add(layers.Dense(hiddenLayersConfig[i], activation=hiddenLayersActivationFunction, name=f"hidden-layer-{i+1}"))
            if withDropOutLayers:
                model.add(layers.Dropout(0.5))
        # output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output-layer'))


    def trainAndSupervise(self, X_train, y_train, X_test, y_test, X_validate, y_validate, epochs=50, earlyStopping=False):
        print("training network with supervised learning")
        print(f"** training set size {len(X_train.index)}")
        print(f"** validation set size {len(X_validate.index)}")
        print(f"** test set size {len(X_test.index)}")
    
        # train and cross-validate (validation set is not used in training ! but added to learning history)
        bestEpoch = -1
        if earlyStopping:
            patience = 50
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto', restore_best_weights=True)

            self.current_run_learning_history = self.nn_model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=epochs, callbacks=[early_stop], batch_size=1)

            bestEpochCandidate = np.argmin(self.current_run_learning_history.history['val_loss'])
            if bestEpochCandidate + patience < epochs: bestEpoch = bestEpochCandidate
        else:

            self.current_run_learning_history = self.nn_model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=epochs, batch_size=1)

        # accuracy when applied on training data after training
        training_scores = self.nn_model.evaluate(X_train, y_train, batch_size=1)
        print("Accuracy for Training Dataset: %.2f%%\n" % (training_scores[1]*100))

        # accuracy when applied on validation data after training
        validation_scores = self.nn_model.evaluate(X_validate, y_validate, batch_size=1)
        print("Accuracy for Validation Dataset: %.2f%%\n" % (validation_scores[1]*100))

        # accuracy when applied on testing data after training
        testing_scores = self.nn_model.evaluate(X_test, y_test, batch_size=1)
        current_run_testing_accuracy = "%.2f%%" % (testing_scores[1]*100)
        print("Accuracy for Testing Dataset: " + current_run_testing_accuracy + "\n")

        # accuracy when applied on all hold-out data (validation + testing data) after training
        X_frames = [X_test, X_validate]
        X_holdout = pd.concat(X_frames, ignore_index=True)
        y_frames = [y_test, y_validate]
        y_holdout = pd.concat(y_frames, ignore_index=True)

        holdout_scores = self.nn_model.evaluate(X_holdout, y_holdout, batch_size=1)
        self.current_run_holdout_accuracy = "%.2f%%" % (holdout_scores[1]*100)
        print("Accuracy for Holdout (Testing & Validation) Dataset: %.2f%%\n" % (holdout_scores[1]*100))

        # compute and return ROC data (and bestEpoch)
        y_holdout_prediction_probabilities = self.nn_model.predict(X_holdout)
        self.current_run_roc_data = roc_curve(y_holdout, y_holdout_prediction_probabilities)

        return self.current_run_roc_data, bestEpoch
    

    def predict(self, pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, pedigree, age, scaler):
        # apply scaling with factors stored in scaler during training
        np_array = np.array([pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, pedigree, age, 0])
        np_array = np_array.reshape(1, -1)
        prognose_X = scaler.transform(np_array)
        prognose_X = np.delete(prognose_X, 8, axis=1)
        
        # call predict
        return self.nn_model.predict(prognose_X)
