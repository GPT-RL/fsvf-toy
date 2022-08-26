# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer-based language models."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.numpy as jnp
from flax import linen as nn
from returns.curry import partial
from returns.pipeline import flow
from supervised.dataset import DataPoint

xavier_uniform = nn.initializers.xavier_uniform()  # type: ignore
normal = nn.initializers.normal(stddev=1e-6)  # type: ignore


@dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    attention_dropout_rate: float = 0.3
    bias_init: Callable = normal
    dropout_rate: float = 0.3
    dtype: Any = jnp.float32
    # emb_dim: int = 16
    # kernel_init: Callable = xavier_uniform
    # mlp_dim: int = 32
    # num_heads: int = 2
    # num_layers: int = 1
    # qkv_dim: int = 64
    emb_dim: int = 512
    kernel_init: Callable = nn.initializers.xavier_uniform()
    mlp_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        """Applies Transformer MlpBlock module."""
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = nn.elu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        output = nn.Dropout(rate=config.dropout_rate)(
            output, deterministic=deterministic
        )
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: input data.
          deterministic: if true dropout is applied otherwise not.

        Returns:
          output after transformer encoder block.
        """
        config = self.config

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.SelfAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=deterministic,
        )(x)

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=config.dtype)(x)
        y = MlpBlock(config=config)(y, deterministic=deterministic)
        return x + y


class Transformer(nn.Module):
    """Transformer Model for sequence tagging."""

    config: TransformerConfig
    dropout_rate: float
    num_actions: int

    @nn.compact
    def __call__(self, *, inputs, train):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          train: if it is training.

        Returns:
          output of a transformer encoder.

        """
        inputs = DataPoint(**inputs)
        b, l, *state_shape = inputs.state.shape
        state = flow(
            inputs.state.reshape(-1, *state_shape),
            nn.Conv(
                features=self.config.emb_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                dtype=jnp.float32,
                padding=0,
            ),
        ).reshape(b, l, -1, self.config.emb_dim)
        action = flow(
            inputs.action.astype(jnp.int32),
            nn.Embed(
                num_embeddings=self.num_actions,
                features=self.config.emb_dim,
                dtype=jnp.int32,
            ),
            partial(jnp.reshape, newshape=(b, l, 1, self.config.emb_dim)),
        )
        value = flow(inputs.value.reshape(b, l, 1, 1), nn.Dense(self.config.emb_dim))
        x = jnp.concatenate([state, action, value], axis=-2)
        _, _, *x_shape = x.shape  # type: ignore
        initializer = nn.initializers.normal()  # type: ignore
        position_embedding = self.param(
            "position_embedding", initializer, (1, *x_shape)
        )

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x + position_embedding
        x = x.reshape(b, -1, self.config.emb_dim)
        x = x[:, :-1]  # exclude target

        for _ in range(self.config.num_layers):
            x = Encoder1DBlock(self.config)(x, deterministic=not train)

        x = nn.LayerNorm(dtype=jnp.float32)(x)
        logits = nn.Dense(1)(x)
        return logits
