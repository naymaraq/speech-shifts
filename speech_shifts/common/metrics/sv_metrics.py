from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from operator import itemgetter

from speech_shifts.common.metrics.metric import Metric
from speech_shifts.common.utils import minimum, maximum

class EqualErrorRate(Metric):
    def __init__(self, prediction_fn=None, name=None):

        self.prediction_fn = prediction_fn
        if name is None:
            name = "EER"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer 
    
    def worst(self, metrics):
        return maximum(metrics)

class DCF(Metric):
    def __init__(self, 
                p_target=0.05,
                c_miss=1.0,
                c_fa=1.0,
                prediction_fn=None, 
                name=None):
        
        self.p_target = p_target
        self.c_miss = c_miss
        self.c_fa = c_fa
        self.prediction_fn = prediction_fn
        if name is None:
            name = "DCF"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        
        fnrs, fprs, thresholds = DCF.compute_error_rates(y_pred, y_true)
        mindcf, threshold = DCF.compute_min_dcf(
        fnrs, fprs, thresholds, self.p_target, self.c_miss, self.c_fa
        )
        return mindcf

    def worst(self, metrics):
        return maximum(metrics)
    
    @staticmethod
    def compute_error_rates(scores, labels):
        sorted_indexes, thresholds = zip(*sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=itemgetter(1)))
        
        labels = [labels[i] for i in sorted_indexes]
        fnrs = []
        fprs = []

        for i in range(0, len(labels)):
            if i == 0:
                fnrs.append(labels[i])
                fprs.append(1 - labels[i])
            else:
                fnrs.append(fnrs[i-1] + labels[i])
                fprs.append(fprs[i-1] + 1 - labels[i])
        fnrs_norm = sum(labels)
        fprs_norm = len(labels) - fnrs_norm

        fnrs = [x / float(fnrs_norm) for x in fnrs]
        fprs = [1 - x / float(fprs_norm) for x in fprs]
        return fnrs, fprs, thresholds

    @staticmethod
    def compute_min_dcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
        min_c_det = float("inf")
        min_c_det_threshold = thresholds[0]
        for i in range(0, len(fnrs)):
            c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det
                min_c_det_threshold = thresholds[i]
        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def
        return min_dcf, min_c_det_threshold