from functools import partial
from re import I

import haiku as hk
import jax
import jax.numpy as jnp

from . import nf

BatchNorm = partial(hk.BatchNorm, decay_rate=0.9, create_offset=True, create_scale=True)


def _depth(x0, d=8):
    x = max(d, int(x0 + d / 2) // d * d)
    if x < 0.9 * x0:
        x += d
    return x


def re(x):
    return jax.nn.relu(x)


def hs(x):
    return jax.nn.hard_swish(x)


LARGE_STEM = [
    # k exp c   se     nl  s
    [3, 16, 16, False, re, 1],
    [3, 64, 24, False, re, 2],
    [3, 72, 24, False, re, 1],
    [5, 72, 40, True, re, 2],
    [5, 120, 40, True, re, 1],
    [5, 120, 40, True, re, 1],
    [3, 240, 80, False, hs, 2],
    [3, 200, 80, False, hs, 1],
    [3, 184, 80, False, hs, 1],
    [3, 184, 80, False, hs, 1],
    [3, 480, 112, True, hs, 1],
    [3, 672, 112, True, hs, 1],
    [5, 672, 160, True, hs, 2],
    [5, 960, 160, True, hs, 1],
    [5, 960, 160, True, hs, 1],
]

SMALL_STEM = [
    # k exp c   se     nl  s
    [3, 16, 16, False, re, 2],
    [3, 72, 24, True, re, 2],
    [3, 88, 24, False, re, 2],
    [8, 96, 40, True, hs, 1],
    [8, 240, 40, True, hs, 2],
    [8, 240, 40, True, hs, 1],
    [8, 120, 48, True, hs, 1],
    [8, 144, 48, True, hs, 1],
    [8, 288, 96, True, hs, 2],
    [8, 576, 96, True, hs, 1],
    [8, 576, 96, True, hs, 1],
    [8, 1, 576, True, hs, 1],
]


class SqueezeExcite(hk.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def __call__(self, x0):
        x = x0.mean(axis=(-3, -2), keepdims=True)
        x = hk.Linear(_depth(x0.shape[-1] * self.ratio))(x)
        x = jax.nn.relu(x)
        x = hk.Linear(x0.shape[-1])(x)
        x = jax.nn.hard_sigmoid(x)
        return x0 * x


class InvertedResidual(hk.Module):
    def __init__(
        self, ch_out, kernel, stride, expand, nl=None, se_ratio=0.25, norm=BatchNorm
    ):
        super().__init__()
        self.ch_out = ch_out
        self.kernel = kernel
        self.stride = stride
        self.expand = expand
        self.activation = nl
        self.se_ratio = se_ratio
        self.normalizer = norm
        self.conv2d = hk.Conv2D if self.normalizer else nf.WSConv2D

    @hk.transparent
    def normalize(self, x, is_training):
        return x if self.normalizer is None else self.normalizer()(x, is_training)

    @hk.transparent
    def activate(self, x, is_training):
        return x if self.activation is None else self.activation(x)

    @hk.transparent
    def squeeze_ex(self, x, is_training):
        if self.se_ratio:
            x = SqueezeExcite(self.se_ratio)(x)
        return x

    @hk.transparent
    def depth_conv(self, x, is_training):
        return hk.DepthwiseConv2D(
            self.expand,
            kernel_shape=self.kernel,
            stride=self.stride,
            with_bias=False,
        )(x)

    @hk.transparent
    def point_conv(self, x, is_training):
        self.conv2d(self.expand, kernel_shape=1, stride=1, with_bias=False)(x)

    def __call__(self, x, is_training):
        for layer in [
            self.point_conv,
            self.normalize,
            self.activate,
            self.depth_conv,
            self.normalize,
            self.activate,
            self.squeeze_ex,
            self.point_conv,
            self.normalize,
        ]:
            ftmaps = layer(x, is_training)
        if self.stride == 1 and x.shape[-1] == self.ch_out:
            return x + ftmaps
        else:
            return ftmaps


class MobileNetV3(hk.Module):
    @hk.transparent
    def preprocess(self, x):
        x = x / 127.5
        x = x - 1.0
        return x

    @hk.transparent
    def avgpool(self, x, keepdims):
        sp_mean = partial(jnp.mean, axis=(-3, -2), keepdims=keepdims)
        ops = (sp_mean,)
        for op in ops:
            x = op(x)
        return x

    def __init__(self, large=True, alpha=1.0, se_ratio=0.25, norm=BatchNorm):
        super().__init__()
        stem_conv = hk.Conv2D if norm else nf.WSConv2D
        self.blocks = [
            self.preprocess,
            stem_conv(_depth(16 * alpha), 3, with_bias=False),
        ]
        if norm:
            self.blocks.append(norm())
        self.blocks.append(hs)

        stem = LARGE_STEM if large else SMALL_STEM
        for k, exp, ch, se, nl, s in stem:
            ch_out = _depth(ch * alpha)
            expand = _depth(exp * alpha)
            if se:
                se = se_ratio
            self.blocks.append(InvertedResidual(ch_out, k, s, expand, nl, se, norm))
        self.blocks += [
            hk.AvgPool(window_shape=7, strides=1, padding="SAME"),
            hk.Conv2D(_depth((1280 if large else 1024) * alpha), 1),
            hs,
        ]
        if se_ratio:
            self.blocks.append(SqueezeExcite(se_ratio))

    def __call__(self, x, is_training=True):
        for block in self.blocks:
            if isinstance(block, InvertedResidual):
                x = block(x, is_training)
            else:
                x = block(x)
        return x
