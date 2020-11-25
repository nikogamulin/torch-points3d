from typing import Dict
import torch
import torchnet as tnt

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.models import model_interface


class SuperquadricsRegressionTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        """ This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch
        Arguments:
            dataset  -- dataset to track (used for the number of classes)
        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(SuperquadricsRegressionTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._loss_dimension = tnt.meter.AverageValueMeter()
        self._loss_epsilon = tnt.meter.AverageValueMeter()
        self._loss_offset = tnt.meter.AverageValueMeter()

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @staticmethod
    def compute_loss_by_components(y_hat, y):
        """ y_hat and y are the predicted and ground-truth tensors of size N x 8
        First three columns represent dimensions along x, y, and z axis.
        Fourth and fifth column represent epsilon_1 and epsilon_2
        Last three columns represent a translation vector.
        """
        dimensions = y[:, 0:3]
        epsilons = y[:, 3:5]
        offsets = y[:, 5:]

        dimensions_hat = y_hat[:, 0:3]
        epsilons_hat = y_hat[:, 3:5]
        offsets_hat = y_hat[:, 5:]

        diff_dimensions = dimensions - dimensions_hat
        avg_loss_dimensions = torch.sum(diff_dimensions*diff_dimensions)/diff_dimensions.numel()

        diff_epsilons = epsilons - epsilons_hat
        avg_loss_epsilons = torch.sum(diff_epsilons*diff_epsilons)/diff_epsilons.numel()

        diff_offsets = offsets - offsets_hat
        avg_loss_offsets = torch.sum(diff_offsets*diff_offsets)/diff_offsets.numel()

        return avg_loss_dimensions, avg_loss_epsilons, avg_loss_offsets


    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = model.get_output()
        # targets = model.get_labels().flatten()
        targets = model.get_labels()

        avg_loss_dimensions, avg_loss_epsilons, avg_loss_offsets = self.compute_loss_by_components(outputs, targets)

        self._loss_dimension.add(avg_loss_dimensions.detach().cpu().numpy())
        self._loss_epsilon.add(avg_loss_epsilons.detach().cpu().numpy())
        self._loss_offset.add(avg_loss_offsets.detach().cpu().numpy())

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_loss_dimension".format(self._stage)] = meter_value(self._loss_dimension)
        metrics["{}__loss_epsilon".format(self._stage)] = meter_value(self._loss_epsilon)
        metrics["{}_loss_offset".format(self._stage)] = meter_value(self._loss_offset)
        return metrics

    @property
    def metric_func(self):
        self._metric_func = {
            "acc": max,
        }  # Those map subsentences to their optimization functions
        return self._metric_func
