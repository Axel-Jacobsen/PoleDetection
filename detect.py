#! /usr/bin/env python3

from collections import namedtuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)


SAVEFILE = 'model.pth'


class PoleDetection(nn.Module):
    """ Model Class for pole detection
    """

    def __init__(self, in_features=5):
        super(PoleDetection, self).__init__()
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.1)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 1)

    def forward(self, x):
        leakyrelu = nn.LeakyReLU()
        sigmoid = nn.Sigmoid()
        x = leakyrelu(self.linear1(x))
        x = leakyrelu(self.linear2(x))
        x = self.dropout(x)
        x = leakyrelu(self.linear3(x))
        x = leakyrelu(self.linear4(x))
        return sigmoid(self.linear5(x))

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()


class TrainModel(object):
    """ Class to train Model
    """

    BATCH_SIZE = 128
    NUM_BATCHES = int(1e5)
    lr = 1e-4

    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def load_data(self):
        """
        Data is List<Tuple<np.ndarray, int>>
        """
        return np.load('pole_data.pkl', allow_pickle=True)

    def split_data(self, data, weight=False):
        """
        Takes prepared data, splits it into training, validation, and test data
        train           70 %
        validation      10 %
        test            20 %
        """
        magnitude = len(data)
        indicies = np.arange(magnitude)
        train_ids = np.random.choice(
                np.array(indicies), size=int(magnitude * 0.7), replace=False
        )
        indicies = np.setdiff1d(indicies, train_ids)
        validation_ids = np.random.choice(
                indicies, size=int(magnitude * 0.1), replace=False
        )
        indicies = np.setdiff1d(indicies, validation_ids)
        test_ids = np.array(indicies)

        training   = torch.as_tensor(data[train_ids])
        validation = torch.as_tensor(data[validation_ids])
        testing    = torch.as_tensor(data[test_ids])

        # Weight training values by making lower frequency labels
        # have higher weighting
        # intended for the y dataset only
        if weight:
            training *= 1 - sum(training) / len(training)

        Data = namedtuple('Data', ['train', 'validation', 'test'])
        return Data(training, validation, testing)

    def prepare_data(self, data):
        """
        Maps over data to create important statistics from data
        """
        xs = np.ndarray(shape=(len(data), 5))
        ys = np.ndarray(shape=(len(data), 1))
        for i, sample in enumerate(data):
            data = sample[0]
            xs[i] = np.array([
                _aspect(data),
                _extent(data),
                _solidity(data),
                _equivalent_diameter(data),
                _orientation(data)
            ])
            ys[i] = sample[1]

        xs_pkt = self.split_data(xs)
        ys_pkt = self.split_data(ys, weight=True)
        return xs_pkt, ys_pkt

    def train(self, plot=False, div=100):
        data = self.load_data()
        x_Data, y_Data = self.prepare_data(data)

        if plot:
            train_results = np.zeros(int(self.NUM_BATCHES / div))
            valid_results = np.zeros(int(self.NUM_BATCHES / div))

        loss = nn.BCELoss()

        j = 0
        for i in range(self.NUM_BATCHES):
            ids = np.random.choice(
                    np.arange(len(x_Data.train)), size=self.BATCH_SIZE, replace=False
            )
            batch_xs = x_Data.train[ids]
            targets  = y_Data.train[ids]
            outs = self.model(batch_xs.float())
            output = loss(outs, targets.float())
            output.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Do some printing/bookkeeping, no actual training here
            if i % div == 0:
                print(f'EPOCH {i} \t LOSS {output.item():.3f}', end='\t')
                with torch.no_grad():
                    if i % (div * 10) == 0:
                        outs = self.model(x_Data.train.float())
                        validation_loss = loss(outs, y_Data.train.float())
                        print(f'VALIDATION LOSS {validation_loss.item():.3f}')
                    else:
                        print()
                    if plot:
                        train_results[j] = output.item()
                        valid_results[j] = validation_loss.item()
                        j += 1
        if plot:
            plt.plot(range(len(train_results)), train_results)
            plt.plot(range(len(valid_results)), valid_results)
            plt.show()

        print(self.accuracy_range(x_Data.test, y_Data.test))

        with torch.no_grad():
            outs = self.model(x_Data.test.float())
            test_loss = loss(outs, y_Data.test.float())
            print(f'TEST LOSS: {test_loss.item():.3f}')
        
        torch.save(self.model.state_dict(), SAVEFILE)

    def accuracy(self, x_data, y_data, threshold=0.1):
        y_output = self.model(x_data.float())
        guesses = y_output > threshold
        correct = guesses.int() & y_data.int()
        return sum(correct) / sum(y_data)

    def accuracy_range(self, x_data, y_data):
        threshold = 0.1
        for i in range(10):
            print(f'threshold: {threshold:.3f} \t accuracy: {self.accuracy(x_data, y_data, threshold=threshold)}')
            threshold += 0.1

def _aspect(data):
    """ Ratio of width to height """
    _, _, w, h = cv.boundingRect(data)
    return w / h


def _extent(data):
    """ Ratio of contour area to bounding rectangle area """
    area = cv.contourArea(data)
    _, _, w, h = cv.boundingRect(data)
    rect_area = w * h
    return area / rect_area


def _solidity(data):
    """ ratio of contour area to convex hull area """
    area = cv.contourArea(data)
    hull = cv.convexHull(data)
    hull_area = cv.contourArea(hull)
    return area / hull_area


def _equivalent_diameter(data):
    area = cv.contourArea(data)
    return np.sqrt(4 * area / np.pi)


def _orientation(data):
    _, _, angle = cv.fitEllipse(data)
    return angle * np.pi / 180


if __name__ == '__main__':
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model = PoleDetection()
    trainer = TrainModel(model)
    try:
        trainer.train(plot=True)
    except KeyboardInterrupt:
        torch.save(trainer.model.state_dict(), SAVEFILE)

