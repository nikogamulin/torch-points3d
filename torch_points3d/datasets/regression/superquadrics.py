import datetime
import time
import pickle

import torch
from tqdm import tqdm
import sys
import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from torch_points3d.metrics.classification_tracker import ClassificationTracker
from torch_points3d.metrics.superquadrics_regression_tracker import SuperquadricsRegressionTracker
from superquadric_generator import Superquadric
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.applications.rsconv import RSConv

sq_offset_range = (25, 230)
sq_dimensions_range = (25, 75)
epsilon_range = (0.1, 1.0)
dimension_max = 305

NUM_WORKERS = 0
BATCH_SIZE = 12

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq

def train_epoch(device):
    model.to(device)
    model.train()
    tracker.reset("train")
    train_loader = dataset.train_dataloader
    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        for i, data in enumerate(tq_train_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            optimizer.zero_grad()
            data.to(device)
            model.forward(data)
            model.backward()
            optimizer.step()
            if i % 10 == 0:
                tracker.track(model)

            tq_train_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()


def test_epoch(device):
    model.to(device)
    model.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    with Ctq(test_loader) as tq_test_loader:
        for i, data in enumerate(tq_test_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            data.to(device)
            model.forward(data)
            tracker.track(model)

            tq_test_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()


class RSConvRegressor(torch.nn.Module):
    def __init__(self, USE_NORMAL):
        super().__init__()
        self.encoder = RSConv("encoder", input_nc=3 * USE_NORMAL, output_nc=8, num_layers=4)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.loss_function = torch.nn.MSELoss()

    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.encoder.conv_type

    def get_output(self):
        """ This is needed by the tracker to get access to the ouputs of the network"""
        return self.output

    def get_labels(self):
        """ Needed by the tracker in order to access ground truth labels"""
        return self.labels

    def get_current_losses(self):
        """ Entry point for the tracker to grab the loss """
        return {"loss": float(self.loss)}

    def forward(self, data):
        # Set labels for the tracker
        self.labels = data.y.squeeze()

        # Forward through the network
        data_out = self.encoder(data)
        self.output = data_out.x.squeeze()

        # Set loss for the backward pass
        # normalize axis and offset values
        labels_normalized = torch.cat([self.labels[:, 0:3]/sq_dimensions_range[1], self.labels[:, 3:5], self.labels[:, 5:]/dimension_max], dim=1)
        output_normalized = torch.cat([self.output[:, 0:3]/sq_dimensions_range[1], self.output[:, 3:5], self.output[:, 5:]/dimension_max], dim=1)
        self.loss = self.loss_function(labels_normalized, output_normalized)

    def backward(self):
        self.loss.backward()


def get_random_superquadric_parameters():
    x0, y0, z0 = random.sample(range(sq_offset_range[0], sq_offset_range[1]), 3)
    a1, a2, a3 = random.sample(range(sq_dimensions_range[0], sq_dimensions_range[1]), 3)
    epsilon1, epsilon2 = list(np.random.uniform(low=epsilon_range[0], high=epsilon_range[1], size=(2,)))
    return a1, a2, a3, epsilon1, epsilon2, x0, y0, z0


class SuperQuadricsRegressionShape(Dataset):
    def __init__(self, dataset_size, points_count, dimension_max, do_normalize=True, transform=None):
        """
        Args:
            number of superquadrics to be gnerated.
        """
        super().__init__()

        used_combinations = []
        self.pos = np.zeros((dataset_size, points_count, 3))
        self.norm = np.zeros((dataset_size, points_count, 3))
        self.y = np.zeros((dataset_size, 8))
        pbar = tqdm(total=dataset_size)
        idx = 0
        while idx < dataset_size:
            combination_is_valid = False
            while not combination_is_valid:
                a1, a2, a3, e1, e2, x0, y0, z0 = get_random_superquadric_parameters()
                current_parameters_str = '{}#{}#{}#{}#{}#{}#{}#{}'.format(a1, a2, a3, e1, e2, x0, y0, z0)
                if current_parameters_str not in used_combinations:
                    combination_is_valid = True
            used_combinations.append(current_parameters_str)
            superquadric = Superquadric(a1, a2, a3, e1, e2, x0, y0, z0, dimension_max)
            try:
                points, normals = superquadric.get_grid(points_count)
                if do_normalize:
                    max_offset = sq_dimensions_range[1] + sq_offset_range[1]
                    points = list(map(lambda x: [item / max_offset for item in x], points))

            except:
                continue
            self.pos[idx] = np.array(points)
            self.norm[idx] = normals
            self.y[idx] = np.array([a1, a2, a3, e1, e2, x0, y0, z0])
            idx += 1
            pbar.update(1)
        pbar.close()

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        norm = torch.from_numpy(self.norm[idx]).float()
        pos = torch.from_numpy(self.pos[idx]).float()
        y = torch.from_numpy(self.y[idx]).double()
        return Data(norm=norm, x=norm, pos=pos, y=y)


class SuperQuadricsRegressionShapeDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = SuperQuadricsRegressionShape(dataset_size=20000, points_count=2048, dimension_max=305, transform=self.train_transform)
        self.test_dataset = SuperQuadricsRegressionShape(dataset_size=4000, points_count=2048, dimension_max=305, transform=self.test_transform)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SuperquadricsRegressionTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


if __name__ == '__main__':
    USE_NORMAL = True  # @param {type:"boolean"}
    DIR = ""
    # TODO: configure your paths
    DATASET_PATH = '/home/niko/Data/superquadrics/sq_dataset_regression_v2.pkl'
    MODELS_PATH = '/home/niko/workspace/torch-points3d/saved_models'
    logdir = "/home/niko/workspace/torch-points3d/runs"
    yaml_config = """
    task: classification
    class: modelnet.ModelNetDataset
    name: modelnet
    dataroot: %s
    pre_transforms:
        - transform: NormalizeScale
        - transform: GridSampling3D
          lparams: [0.02]
    train_transforms:
        - transform: FixedPoints
          lparams: [2048]
    test_transforms:
        - transform: FixedPoints
          lparams: [2048]
    """ % (os.path.join(DIR, "data"))

    from omegaconf import OmegaConf
    params = OmegaConf.create(yaml_config)
    if os.path.isfile(DATASET_PATH):
        with open(DATASET_PATH, 'rb') as input:
            dataset = pickle.load(input)
    else:
        dataset = SuperQuadricsRegressionShapeDataset(params)
        with open(DATASET_PATH, 'wb') as output:
            pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
    # Setup the data loaders
    model = RSConvRegressor(USE_NORMAL)
    dataset.create_dataloaders(
        model,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        precompute_multi_scale=False
    )
    print(next(iter(dataset.test_dataloaders[0])))
    # Setup the tracker and actiavte tensorboard loging
    
    logdir = os.path.join(logdir, str(datetime.datetime.now()))
    os.mkdir(logdir)
    os.chdir(logdir)
    tracker = dataset.get_tracker(False, True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    EPOCHS = 200
    for i in range(EPOCHS):
        print("=========== EPOCH %i ===========" % i)
        time.sleep(0.5)
        train_epoch('cuda')
        tracker.publish(i)
        test_epoch('cuda')
        tracker.publish(i)
        if i % 50 == 0:
            torch.save(model.state_dict(), f'{MODELS_PATH}/regression_{i}_dict_model.pt')