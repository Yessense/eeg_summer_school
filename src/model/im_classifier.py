from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn


class EnvelopeDetector(nn.Module):
    def __init__(self, in_channels: int, channels_multiplier: int, filtering_s):
        super(EnvelopeDetector, self).__init__()
        # TODO: Channels multiplier? Различные фильтры
        self.channels_multiplier = channels_multiplier

        self.hidden_channels = self.channels_multiplier * in_channels
        self.filtering_size = 100
        # TODO: Envelope size? Длина bandpass фильтра
        self.envelope_size = 50

        # Layers
        self.conv_filtering = nn.Conv1d(in_channels, self.hidden_channels,
                                        kernel_size=self.filtering_size, groups=self.hidden_channels, bias=False)
        self.pre_envelope_batchnorm = nn.BatchNorm1d(self.hidden_channels, affine=False)
        self.conv_envelope = nn.Conv1d(in_channels, self.hidden_channels,
                                       kernel_size=self.filtering_size, groups=self.hidden_channels)
        # TODO: Grad? Всегда складывает
        self.conv_envelope.requires_grad = False
        # TODO: Weight data? Нормированная сумма
        self.conv_envelope.weight.data = (
                torch.ones(self.hidden_channels * self.envelope_size) / self.filtering_size
        ).reshape((self.hidden_channels, 1, self.envelope_size))

    def forward(self, x):
        x = self.conv_filtering(x)
        x = self.pre_envelope_batchnorm(x)
        x = torch.abs(x)
        x = self.conv_envelope(x)
        return x


class IMClassifier(pl.LightningModule):
    def __init__(self, in_channels, n_classes, lag_backward, channels_multiplier=1):
        super(IMClassifier, self).__init__()
        # TODO: Pre out?
        self.pointwise_out = 3
        self.fin_layer_decim = 20
        self.channels_multiplier = channels_multiplier
        # TODO: Lag backward?
        self.lag_backward = lag_backward

        # Layers
        self.pointwise_conv = nn.Conv1d(in_channels, self.pointwise_out, kernel_size=1)
        self.pointwise_bn = torch.nn.BatchNorm1d(self.pointwise_out, affine=False)

        self.detector = EnvelopeDetector(self.pointwise_out, self.channels_multiplier)
        # TODO: detector-out? lag_backward = длина окна, длина окна фильтра, длина окна фильтра огибающей
        # И прореживаем
        self.detector_out = self.pointwise_out * (
                (lag_backward - self.detector.filtering_size - self.detector.envelope_size + 2)
                // self.fin_layer_decim)
        self.detector_bn = nn.BatchNorm1d(self.detector_out, affine=False)

        self.classifier = nn.Linear(self.detector_out, n_classes)

    def forward(self, x):
        inputs = self.pointwise_conv(x)
        inputs = self.pointwise_bn(inputs)

        detected_envelopes = self.detector(inputs)

        left_samples_slice = slice(((
                                                self.lag_backward - self.detector.filtering_size - self.detector.envelope_size + 2) % self.fin_layer_decim),
                                   None, self.fin_layer_decim)
        features = detected_envelopes[:, :, left_samples_slice]

        output = self.classifier(features)
        return output

# Задние висят и много шума
# Боковые - мышцы, передние - глаза.


# Запустить бейзлайн
# Трейн
# Тест 30 человек
# accuracy prescision recall
# Второй, поиграться с параметрами на других данных
# 1 - ансамбль
# 2 -
