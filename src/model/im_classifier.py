from argparse import ArgumentParser
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy


class EnvelopeDetector(nn.Module):
    def __init__(self, in_channels: int,
                 channels_multiplier: int = 1,
                 bandpass_filter_size: int = 100,
                 lowpass_filter_size: int = 50):
        super(EnvelopeDetector, self).__init__()

        self.channels_multiplier = channels_multiplier  # Channels multiplier? Различные фильтры
        self.bandpass_filter_size = bandpass_filter_size
        self.lowpass_filter_size = lowpass_filter_size

        # Layers
        self.hidden_channels = self.channels_multiplier * in_channels
        self.conv_filtering = nn.Conv1d(in_channels, self.hidden_channels,
                                        kernel_size=self.bandpass_filter_size, groups=self.hidden_channels, bias=False)
        self.pre_envelope_batchnorm = nn.BatchNorm1d(self.hidden_channels, affine=False)

        # Weight data = Нормированная сумма
        self.conv_envelope = nn.Conv1d(in_channels, self.hidden_channels,
                                       kernel_size=self.bandpass_filter_size, groups=self.hidden_channels)
        self.conv_envelope.requires_grad = False
        self.conv_envelope.weight.data = (
                torch.ones(self.hidden_channels * self.lowpass_filter_size) / self.bandpass_filter_size
        ).reshape((self.hidden_channels, 1, self.lowpass_filter_size))
        self.activation = torch.abs

    def forward(self, x):
        x = self.conv_filtering(x)
        x = self.pre_envelope_batchnorm(x)
        x = self.activation(x)
        x = self.conv_envelope(x)
        return x


class IMClassifier(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("IMClassifier")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--in_channels", type=int, default=27)
        parser.add_argument("--n_classes", type=int, default=3)
        parser.add_argument("--lag_backward", type=int, default=256)
        parser.add_argument("--pointwise_out", type=int, default=3)
        parser.add_argument("--fin_layer_decim", type=int, default=20)
        parser.add_argument("--bandpass_filter_size", type=int, default=100)
        parser.add_argument("--lowpass_filter_size", type=int, default=50)
        return parent_parser

    def __init__(self, in_channels,
                 n_classes: int = 3,
                 lag_backward: int = 256,  # Ширина окна
                 channels_multiplier: int = 1,  # Раздувание каналов в EnvelopeDetector
                 lr: float = 3e-4,
                 pointwise_out: int = 3,  # Количество каналов после первой свертки
                 fin_layer_decim: int = 20,  # Прореживание
                 **kwargs):
        super(IMClassifier, self).__init__()
        self.pointwise_out = pointwise_out
        self.fin_layer_decim = fin_layer_decim
        self.channels_multiplier = channels_multiplier
        self.lr = lr
        self.lag_backward = lag_backward

        # Layers
        self.pointwise_conv = nn.Conv1d(in_channels, self.pointwise_out, kernel_size=1)
        self.pointwise_bn = torch.nn.BatchNorm1d(self.pointwise_out, affine=False)

        self.detector = EnvelopeDetector(self.pointwise_out,
                                         self.channels_multiplier,
                                         bandpass_filter_size=kwargs['bandpass_filter_size'],
                                         lowpass_filter_size=kwargs['lowpass_filter_size'])
        # TODO: detector-out lag_backward = длина окна, длина окна фильтра, длина окна фильтра огибающей

        # N most recent samples
        # self.detector_out = self.pointwise_out * (
        #         (lag_backward - self.detector.bandpass_filter_size - self.detector.lowpass_filter_size + 2)
        #         // self.fin_layer_decim)
        # self.detector_bn = nn.BatchNorm1d(self.detector_out, affine=False)

        self.detector_out = 1090
        self.dropout = nn.Dropout(p=0.9)
        self.classifier = nn.Linear(self.detector_out, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.accuracy = Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        # Spatial filtering
        inputs = self.pointwise_conv(x)
        inputs = self.pointwise_bn(inputs)

        # Adaptive envelope extractor
        detected_envelopes = self.detector(inputs)
        # TODO: Detector batch norm???

        # N most recent samples
        # start = self.lag_backward - self.detector.bandpass_filter_size - self.detector.lowpass_filter_size + 2
        # left_samples_slice = slice(start % self.fin_layer_decim, None, self.fin_layer_decim)
        features = detected_envelopes  # [:, :, left_samples_slice].contiguous()
        features = features.view(features.size(0), -1)
        # features = self.detector_bn(features)
        features = self.dropout(features)

        output = self.classifier(features)
        output = self.sigmoid(output)

        return output

    def test_step(self, batch, idx):
        data, y_target = batch
        y_predicted = self.forward(data)

        loss = self.loss_func(y_predicted, y_target)
        accuracy = self.accuracy(y_predicted, y_target)
        self.log("Test Loss", loss)
        self.log("Test Accuracy", accuracy, prog_bar=True)

    def validation_step(self, batch, idx, dataloader_idx):
        data, y_target = batch

        y_predicted = self.forward(data)

        loss = self.loss_func(y_predicted, y_target)
        accuracy = self.accuracy(y_predicted, y_target)

        if dataloader_idx == 0:
            self.log("Val Loss", loss)
            self.log("Val Accuracy", accuracy, prog_bar=True)
        if dataloader_idx == 1:
            self.log("Test Loss", loss)
            self.log("Test Accuracy", accuracy, prog_bar=True)

    def training_step(self, batch):
        data, y_target = batch

        y_predicted = self.forward(data)

        loss = self.loss_func(y_predicted, y_target)
        accuracy = self.accuracy(torch.argmax(y_predicted, dim=1), y_target)

        self.log("Train Loss", loss)
        self.log("Train Accuracy", accuracy, prog_bar=True)

        return loss

    def loss_func(self, y_pred, y_true):
        loss = nn.CrossEntropyLoss()
        return loss(y_pred, y_true)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

# Задние висят и много шума
# Боковые - мышцы, передние - глаза.


# Запустить бейзлайн
# Трейн
# Тест 30 человек
# accuracy prescision recall
# Второй, поиграться с параметрами на других данных
# 1 - ансамбль
# 2 -


# Симметричность весов в конволюционных слоях
# Отражение ряда в батче для рандомных индексов
# Предзадать коэффициенты в lowpass фильтре
# Инициализировать веса первого слоя, найдя коэффициенты фильтра csp
