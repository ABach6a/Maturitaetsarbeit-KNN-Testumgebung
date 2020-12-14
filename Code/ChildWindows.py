from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtGui as qtg 
from PyQt5 import QtCore as qtc
from CanvasWidgets import HistogramWidget
from CanvasWidgets import AccuracyWidget


class LearningWindow(qtw.QWidget):

    """LearningWindow constructor"""
    def __init__(self, networkManager): 
        super().__init__()
        self._run_history = networkManager.current_run_learning_history
        if not networkManager.current_run_topology:
            self.setWindowTitle("Learning Graph for 'No hidden Layers'-Run")
        else:
            self.setWindowTitle("Learning Graph for " + str(networkManager.current_run_topology) + " - Run")
        #self.resize(1000, 800)
        mainLayout = qtw.QVBoxLayout()
        mainLayout.addWidget(AccuracyWidget(self._run_history))
        self.setLayout(mainLayout)

              
class HistogrammWindow(qtw.QWidget):

    """HistogramWindow constructor"""
    def __init__(self, dataManager):
        super().__init__()
        self.setWindowTitle("Histogramm")
        self.resize(1000, 800)
        mainLayout = qtw.QVBoxLayout()
        mainLayout.addWidget(HistogramWidget(dataManager.odf, width=20, height=10, dpi=50))
        self.setLayout(mainLayout)


class TabelleWindow(qtw.QWidget):

    # _dataManager = None : nicht nötig, weil nur im Konstruktor gebraucht

    """TabelleWindow constructor"""
    def __init__(self, dataManager):
        super().__init__()
        self.setWindowTitle("Tabelle")
        self.resize(970, 600)
        mainLayout = qtw.QVBoxLayout()
        tableview = qtw.QTableView()
        tableview.setModel(PandasModel(dataManager.odf))
        mainLayout.addWidget(tableview)
        self.setLayout(mainLayout)

class PandasModel(qtc.QAbstractTableModel):
    def __init__(self, data, parent=None):
        qtc.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=qtc.Qt.DisplayRole):
        if index.isValid():
            if role == qtc.Qt.DisplayRole:
                return qtc.QVariant(str(
                    self._data.values[index.row()][index.column()]))
        return qtc.QVariant()

    def headerData(self, section, orientation, role):
        if (
         orientation == qtc.Qt.Horizontal and 
         role == qtc.Qt.DisplayRole
         ): 
            return self._data.columns[section]
        else: 
            return super().headerData(section, orientation, role)   
