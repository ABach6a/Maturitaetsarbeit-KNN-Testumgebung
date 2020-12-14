import sys 
from NeuralNetworkManagement import NNManager
from PimaDataManagement import DataManager
from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtGui as qtg 
from PyQt5 import QtCore as qtc
from QLed import QLed
from matplotlib import pyplot as plt 
from ChildWindows import LearningWindow
from ChildWindows import TabelleWindow
from ChildWindows import HistogrammWindow
from ChildWindows import LearningWindow
from CanvasWidgets import ROCWidget
import sklearn.metrics as metrics

class MainWindow(qtw.QWidget):

    _neuralNetworkManager = NNManager()
    _dataManager = DataManager()
    _rocWidget = ROCWidget()
    _childWindows = list()

    """MainWindow constructor"""
    def __init__(self): 
        super().__init__()

        stylesheet = """
        QGroupBox {
            font-family: "Edwardian Script ITC";
            font-size: 40px;
            font-weight: bold;
            color: #8B0000
        }"""
        self.setStyleSheet(stylesheet)

        # Main UI code goes here
        self.setWindowTitle("Deep Learning Testumgebung für Diabetesvorhersage bei den Pima Indianern")
        self.resize(800, 900)
    
        mainLayout = qtw.QVBoxLayout()

        # group box "Trainieren"
        groupBox_trainieren = qtw.QGroupBox("Trainieren")
        groupboxLayout = qtw.QHBoxLayout()
        groupBox_trainieren.setLayout(groupboxLayout)

        mainLayout.addWidget(groupBox_trainieren)
        self.populateGUISectionOne(groupboxLayout)

        # group box "Daten"
        groupBox_Daten = qtw.QGroupBox("Daten")
        groupboxLayout = qtw.QHBoxLayout()
        groupBox_Daten.setLayout(groupboxLayout)

        mainLayout.addWidget(groupBox_Daten)
        self.populateGUISectionTwo(groupboxLayout)

        # group box "Roc Kurven"
        groupBox_Roc = qtw.QGroupBox("ROC Kurven")
        groupboxLayout = qtw.QHBoxLayout()
        groupBox_Roc.setLayout(groupboxLayout)

        mainLayout.addWidget(groupBox_Roc)
        self.populateGUISectionThree(groupboxLayout)
        
        # group box "Prognose"
        groupBox_Prognose = qtw.QGroupBox("Prognose")
        groupboxLayout = qtw.QHBoxLayout()
        groupBox_Prognose.setLayout(groupboxLayout)

        mainLayout.addWidget(groupBox_Prognose)
        self.populateGUISectionFour(groupboxLayout)

        self.setLayout(mainLayout)
        # End main UI code 
        self.show()

    # GUI Section One: Trainieren 

    def populateGUISectionOne(self, layout):
        # topology 
        self.textfield_topology = qtw.QLineEdit()
        self.textfield_topology.setPlaceholderText("Hidden Layers Topology")
        layout.addWidget(self.textfield_topology)

        configLayout = qtw.QVBoxLayout()
        
        self.textfield_epochs = qtw.QLineEdit()
        self.textfield_epochs.setPlaceholderText("epochs = 50")

        labelOptionLayout = qtw.QHBoxLayout()
        self.linear_checkBox = qtw.QCheckBox('use linear activation')
        labelOptionLayout.addWidget(self.linear_checkBox)

        labelDropoutLayout = qtw.QHBoxLayout()
        self.dropout_checkBox = qtw.QCheckBox('add dropout layers')
        labelDropoutLayout.addWidget(self.dropout_checkBox)
        self.earlyStopping_checkBox = qtw.QCheckBox('stop early, grace 50')
        labelDropoutLayout.addWidget(self.earlyStopping_checkBox)
        
        configLayout.addLayout(labelOptionLayout)
        configLayout.addWidget(self.textfield_epochs)
        configLayout.addLayout(labelDropoutLayout)

        layout.addLayout(configLayout)
        
        button_train = qtw.QPushButton("Trainieren")
        layout.addWidget(button_train)
        
        # light
        self.led=QLed(self, onColour=QLed.Grey, shape=QLed.Circle)
        self.led.value = True
        layout.addWidget(self.led)

        # Lenrbericht button
        button_lb = qtw.QPushButton("Lernbericht")
        layout.addWidget(button_lb)

        # widget connections 
        button_train.clicked.connect(self.trainButtonClicked)
        button_lb.clicked.connect(self.buttonBerichtClicked)

    def buttonBerichtClicked(self):
        self.changeLEDColor(QLed.Blue)
        learningWindow = LearningWindow(self._neuralNetworkManager)
        self._childWindows.append(learningWindow)
        learningWindow.show()

    def trainButtonClicked(self):
        self.changeLEDColor(QLed.Orange)
        
        self._dataManager.splitDataIntoTrainingValidationAndTestingSets()

        # create the neural network
        hiddenLayersConfig, epochs, linear, dropout, earlyStopping= self._getTrainingParameters()
        self._neuralNetworkManager.createNetworkModel(hiddenLayersConfig, not linear, dropout)

        # train the neural network
        roc_data, bestEpoch = self._neuralNetworkManager.trainAndSupervise(
            self._dataManager.X_train, self._dataManager.y_train, 
            self._dataManager.X_test, self._dataManager.y_test, 
            self._dataManager.X_validate, self._dataManager.y_validate,
            int(epochs), earlyStopping)

        self.changeLEDColor(QLed.Green)

        # compute AUC
        FPR, TPR, thresholds = roc_data
        roc_auc = metrics.auc(FPR, TPR)

        if bestEpoch != -1: epochs = bestEpoch
        
        # compose labels
        roc_auc_as_string = "%.2f" % roc_auc
        if not hiddenLayersConfig:
            label = "no hidden layers e(" + str(epochs) 
        else:
            label = str(hiddenLayersConfig) + " e(" + str(epochs)
        
        if linear:
            label = label + ")/l -> acc(0.50, "
        else:
            if dropout:
                label = label + ")/d -> acc(0.50, "
            else:
                label = label + ") -> acc(0.50, "

        label = label + self._neuralNetworkManager.current_run_holdout_accuracy + ") / AUC(" + roc_auc_as_string + ")"

        # plot ROC
        self._rocWidget.plot(FPR, TPR, thresholds, label)


    def _getTrainingParameters(self):
        try:
            hiddenLayersConfig = self.convertStringToListOfInteger(self.textfield_topology.text())
        except ValueError as e:
            self.changeLEDColor(QLed.Red)
            print(f"### !! please enter integers separated by comma ONLY !! ###: {e}")
            return
        epochs = self.textfield_epochs.text()
        if not epochs:
            epochs = 50

        return hiddenLayersConfig, epochs, self.linear_checkBox.isChecked(), self.dropout_checkBox.isChecked(), self.earlyStopping_checkBox.isChecked()

    def changeLEDColor(self, color):
        self.led.setOnColour(color)
        # print("forcing repaint")
        self.repaint()
    
    def convertStringToListOfInteger(self,topology : str): 
        if (len(topology) == 0): return []
        else: return [int(s) for s in topology.split(',')]

    # ... end GUI Section One

    # GUI Section Two: Daten 

    def populateGUISectionTwo(self,layout):
        # topology
        self.button_tabelle = qtw.QPushButton("Tabelle")
        layout.addWidget(self.button_tabelle)
        self.button_histogramm = qtw.QPushButton("Histogramm")
        layout.addWidget(self.button_histogramm)

        # widget connection 
        self.button_histogramm.clicked.connect(self.buttonHistogrammClicked)
        self.button_tabelle.clicked.connect(self.buttonTabelleClicked)

    def buttonHistogrammClicked(self):
        histogrammWindow = HistogrammWindow(self._dataManager)
        self._childWindows.append(histogrammWindow)
        histogrammWindow.show()

    def buttonTabelleClicked(self):
        tabelleWindow = TabelleWindow(self._dataManager)
        self._childWindows.append(tabelleWindow)
        tabelleWindow.show()

    # ... end GUI Section Two

    # GUI Section Three: Roc Kurven 

    def populateGUISectionThree(self, layout):
        layout.addWidget(self._rocWidget)

    # ... end GUI Section Three

    # GUI Section Four: Prognose

    def populateGUISectionFour(self, layout):
        self.textfield_pregnancies = qtw.QLineEdit()
        self.textfield_pregnancies.setPlaceholderText("Pregnancies")
        self.textfield_glucose = qtw.QLineEdit()
        self.textfield_glucose.setPlaceholderText("Glucose")
        self.textfield_bloodPressure = qtw.QLineEdit()
        self.textfield_bloodPressure.setPlaceholderText("BloodPressure")
        self.textfield_skinThickness = qtw.QLineEdit()
        self.textfield_skinThickness.setPlaceholderText("SkinThickness")
        self.textfield_insulin = qtw.QLineEdit()
        self.textfield_insulin.setPlaceholderText("Insulin")
        self.textfield_bmi = qtw.QLineEdit()
        self.textfield_bmi.setPlaceholderText("BMI")
        self.textfield_pedigree = qtw.QLineEdit()
        self.textfield_pedigree.setPlaceholderText("Pedigree")
        self.textfield_age = qtw.QLineEdit()
        self.textfield_age.setPlaceholderText("Age")

        verticalGroupboxLayoutRows = qtw.QVBoxLayout()
        verticalGroupboxLayoutPrognose = qtw.QVBoxLayout()
        
        horizontalFirstRowLayout = qtw.QHBoxLayout()
        horizontalSecondRowLayout = qtw.QHBoxLayout()

        horizontalFirstRowLayout.addWidget(self.textfield_pregnancies)
        horizontalFirstRowLayout.addWidget(self.textfield_glucose)
        horizontalFirstRowLayout.addWidget(self.textfield_bloodPressure)
        horizontalFirstRowLayout.addWidget(self.textfield_skinThickness)
        horizontalSecondRowLayout.addWidget(self.textfield_insulin)
        horizontalSecondRowLayout.addWidget(self.textfield_bmi)
        horizontalSecondRowLayout.addWidget(self.textfield_pedigree)      
        horizontalSecondRowLayout.addWidget(self.textfield_age)   

        verticalGroupboxLayoutRows.addLayout(horizontalFirstRowLayout)
        verticalGroupboxLayoutRows.addLayout(horizontalSecondRowLayout)

        button_prognose = qtw.QPushButton("Prognose")
        self.textfield_prognose = qtw.QLineEdit()
        self.textfield_prognose.setReadOnly(True)

        horizontalPrognoseLayout = qtw.QHBoxLayout()
        horizontalPrognoseLayout.addWidget(button_prognose)
        horizontalPrognoseLayout.addWidget(self.textfield_prognose)

        verticalGroupboxLayoutPrognose.addLayout(horizontalPrognoseLayout)

        layout.addLayout(verticalGroupboxLayoutRows)
        layout.addLayout(verticalGroupboxLayoutPrognose)
        
        # widget connection 
        button_prognose.clicked.connect(self.buttonPrognoseClicked)

    def buttonPrognoseClicked(self):
        result = self._neuralNetworkManager.predict(
            self.textfield_pregnancies.text(),
            self.textfield_glucose.text(),
            self.textfield_bloodPressure.text(),
            self.textfield_skinThickness.text(),
            self.textfield_insulin.text(),
            self.textfield_bmi.text(),
            self.textfield_pedigree.text(),      
            self.textfield_age.text(),
            self._dataManager.scaler
        )
        self.textfield_prognose.setText("%.2f" % result)

    # ... end GUI Section Four

    def closeEvent(self, event):
        for childWindow in self._childWindows:
            childWindow.close()   


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec())
