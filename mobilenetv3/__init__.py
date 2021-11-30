from functools import partial

import haiku
import jax
import jax.numpy as jnp
import numpy as np


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


class SqueezeExcite(haiku.Module):
    def __init__(self, reduction=4):
        super().__init__()
        self.reduction = reduction

    def __call__(self, x):
        d = x.shape[-1]
        channels = x.mean(axis=(-3, -2), keepdims=True)
        squeezed = haiku.Linear(d // self.reduction)(channels)
        attended = haiku.Linear(d)(squeezed)
        attn_nrm = jax.nn.relu6(attended + 3) / 6
        return x * attn_nrm


class InvertedResidual(haiku.Module):
    def __init__(
        self,
        ch_out,
        kernel,
        stride,
        expand,
        nl=None,
        use_se=False,
        normalizer=haiku.BatchNorm,
    ):
        super().__init__()
        self.ch_out = ch_out
        self.use_se = use_se
        self.kernel = kernel
        self.stride = stride
        self.expand = expand
        self.activation = nl
        self.normalizer = normalizer

    @haiku.transparent
    def normalize(self, x):
        return x if self.normalizer is None else self.normalizer(x)

    @haiku.transparent
    def activate(self, x):
        return x if self.activation is None else self.activation(x)

    @haiku.transparent
    def squeeze_ex(self, x):
        return SqueezeExcite()(x) if self.use_se else x

    @haiku.transparent
    def depth_conv(self, x):
        return haiku.Conv2D(
            x.shape[-1],
            kernel_shape=self.kernel,
            stride=self.stride,
            with_bias=False,
            feature_group_count=self.expand,
        )(x)

    @haiku.transparent
    def point_conv(self, x, d=None):
        haiku.Conv2D(d or self.expand, kernel_shape=1, stride=1, with_bias=False)(x)

    def __call__(self, x):
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
            ftmaps = layer(x)
        if self.stride == 1 and x.shape[-1] == self.ch_out:
            return x + ftmaps
        else:
            return ftmaps


class MobileNetV3(haiku.Module):
    @haiku.transparent
    def preprocess(self, x):
        x = x / 127.5
        x = x - 1.0
        return x

    @haiku.transparent
    def avgpool(self, x, keepdims):
        sp_mean = partial(jnp.mean, axis=(-3, -2), keepdims=keepdims)
        ops = (sp_mean,)
        for op in ops:
            x = op(x)
        return x

    def __init__(self, large=True, normalizer=haiku.BatchNorm, alpha=1.0):
        super().__init__()

        def make_divisible(n, d=8):
            return int(np.ceil(n * 1.0 / d) * d)

        self.blocks = [self.preprocess, haiku.Conv2D(16, 3, with_bias=False), hs]

        stem = LARGE_STEM if large else SMALL_STEM
        for k, exp, ch, se, nl, s in stem:
            ch_out = make_divisible(ch * alpha)
            expand = make_divisible(exp * alpha)
            self.blocks.append(
                InvertedResidual(ch_out, k, s, expand, nl, se, normalizer)
            )
        self.blocks += [
            haiku.AvgPool(window_shape=7),
            haiku.Conv2D(make_divisible((1280 if large else 1024) * alpha), 1),
            hs,
            SqueezeExcite(),
        ]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x
