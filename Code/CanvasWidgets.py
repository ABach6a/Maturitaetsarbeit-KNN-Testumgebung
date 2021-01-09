from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class HistogramWidget(FigureCanvas):

    def __init__(self, df, width = 5, height = 5, dpi = 200):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.figure)
        self._df = df
        self.plot()

    def plot(self):
        ax = self.figure.add_subplot(111)
        self._df.hist(ax=ax, bins=10)
        ax.plot()
        
class ROCWidget(FigureCanvas):

    _figure = None

    def __init__(self, width = 8, height = 8):
        self._figure = Figure(figsize=(width, height))
        FigureCanvas.__init__(self, self._figure)
        self.axes = self.figure.add_subplot(111)
        self._plotFrame(self.axes)

    def plot(self, falsePositiveRateValues, truePositiveRateValues, thresholds, legendLabel):
        p = self.axes.plot(falsePositiveRateValues, truePositiveRateValues, label=legendLabel)
        self.axes.legend(loc="lower right")
        self._plotThreshold(falsePositiveRateValues, truePositiveRateValues, thresholds, p[-1].get_color())
        self.draw()
    
    def _plotThreshold(self, falsePositiveRateValues, truePositiveRateValues, thresholds, color):
        # plot threshold in the middle of the roc data
        for i in range(len(thresholds)):
            if thresholds[i] <= 0.5:
                break
        indexBelow = i
        indexAbove = i-1
        self.axes.plot(falsePositiveRateValues[indexBelow], truePositiveRateValues[indexBelow], 'o-', color=color)
        self.axes.annotate(np.round(thresholds[indexBelow],2), (falsePositiveRateValues[indexBelow], truePositiveRateValues[indexBelow]-0.04), color=color)    
        self.axes.plot(falsePositiveRateValues[indexAbove], truePositiveRateValues[indexAbove], 'o-', color=color)
        self.axes.annotate(np.round(thresholds[indexAbove],2), (falsePositiveRateValues[indexAbove], truePositiveRateValues[indexAbove]-0.04), color=color)   

    def _plotFrame(self, ax):
        ax.grid(True)
        #ax.set_title('ROC curves')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        self._plotRandomGuessBaseline(ax)

    def _plotRandomGuessBaseline(self, ax):
        ax.plot([0,1],[0,1], 'k--', linewidth=.5, color='black') #diagonal line


class AccuracyWidget(FigureCanvas):
    
    def __init__(self, runHistory, width = 8, height = 8):
        self._figure = Figure(figsize=(width, height))
        self.run_history = runHistory
        FigureCanvas.__init__(self, self._figure)
        self.axes = self.figure.add_subplot(111)
        self._plotFrame(self.axes)
        self._plotRunHistoryAccuracies(self.axes)

    def _plotFrame(self, ax):
        ax.grid(True)
        ax.set_title('Learning progress')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')

    def _plotRunHistoryAccuracies(self, ax):
        ax.plot(self.run_history.history['loss'], 'blue', marker='.', label='training-data loss')
        ax.plot(self.run_history.history['val_loss'], 'red', marker='.', label='validation-data loss')
        ax.legend()
