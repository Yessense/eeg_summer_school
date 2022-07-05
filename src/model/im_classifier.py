from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn


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

    def forward(self, x):
        x = self.conv_filtering(x)
        x = self.pre_envelope_batchnorm(x)
        x = torch.abs(x)
        x = self.conv_envelope(x)
        return x


class IMClassifier(pl.LightningModule):
    def __init__(self, in_channels, n_classes, lag_backward, channels_multiplier=1):
        super(IMClassifier, self).__init__()
        self.pointwise_out = 3
        self.fin_layer_decim = 20
        self.channels_multiplier = channels_multiplier
        # window size
        self.lag_backward = lag_backward

        # Layers
        self.pointwise_conv = nn.Conv1d(in_channels, self.pointwise_out, kernel_size=1)
        self.pointwise_bn = torch.nn.BatchNorm1d(self.pointwise_out, affine=False)

        self.detector = EnvelopeDetector(self.pointwise_out, self.channels_multiplier)
        # TODO: detector-out lag_backward = длина окна, длина окна фильтра, длина окна фильтра огибающей

        # N most recent samples
        self.detector_out = self.pointwise_out * (
                (lag_backward - self.detector.bandpass_filter_size - self.detector.lowpass_filter_size + 2)
                // self.fin_layer_decim)
        self.detector_bn = nn.BatchNorm1d(self.detector_out, affine=False)

        self.classifier = nn.Linear(self.detector_out, n_classes)

    def forward(self, x):
        # Spatial filtering
        inputs = self.pointwise_conv(x)
        inputs = self.pointwise_bn(inputs)

        # Adaptive envelope extractor
        detected_envelopes = self.detector(inputs)
        # TODO: Detector batch norm???

        # N most recent samples
        start = self.lag_backward - self.detector.bandpass_filter_size - self.detector.lowpass_filter_size + 2
        left_samples_slice = slice(start % self.fin_layer_decim, None, self.fin_layer_decim)
        features = detected_envelopes[:, :, left_samples_slice].contiguous()
        features = features.view(features.size(0), -1)
        output = self.classifier(features)

        return output

    def training_step(self, batch):
        X_batch, y_batch, batch_idx = batch
        y_batch = y_batch.argmax(axis=1)

        X_batch = X_batch.float()
        y_batch = y_batch.long()
        y_predicted = self.forward(X_batch)

        loss = self.loss_func(y_predicted, y_batch)

        return loss

    def loss_func(self, y_pred, y_true):
        loss = nn.CrossEntropyLoss()
        return loss(y_pred, y_true)


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=3e-4)
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
