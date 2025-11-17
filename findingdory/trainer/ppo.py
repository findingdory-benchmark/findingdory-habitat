#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.utils.common import LagrangeInequalityCoefficient
from habitat_baselines.rl.ppo.updater import Updater


@baseline_registry.register_updater
class ImageNavPPO(PPO):
    '''
    Same as default PPO but allows different LR for visual encoder parameters + weight decay
    '''

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config):
        return cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            encoder_lr=config.encoder_lr,
            wd=config.wd,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
        )

    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        encoder_lr: Optional[float] = None,
        wd: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        Updater.__init__(self)

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.device = next(actor_critic.parameters()).device

        if (
            use_adaptive_entropy_pen
            and hasattr(self.actor_critic, "num_actions")
            and getattr(self.actor_critic, "action_distribution_type", None)
            == "gaussian"
        ):
            num_actions = self.actor_critic.num_actions

            self.entropy_coef = LagrangeInequalityCoefficient(
                -float(entropy_target_factor) * num_actions,
                init_alpha=entropy_coef,
                alpha_max=1.0,
                alpha_min=1e-4,
                greater_than=True,
            ).to(device=self.device)

        self.use_normalized_advantage = use_normalized_advantage
        self.optimizer = self._create_optimizer(lr, eps, encoder_lr, wd)

        self.non_ac_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("actor_critic.")
        ]

    def _create_optimizer(self, lr, eps, encoder_lr, wd):
        
        # Separate out the visual encoder parameters from the rest of the policy parameters
        visual_encoder_params, other_params = [], []
        for name, param in self.actor_critic.named_parameters():
            if param.requires_grad:
                if (
                    "net.visual_encoder.backbone" in name
                    or "net.goal_visual_encoder.backbone" in name
                ):
                    visual_encoder_params.append(param)
                else:
                    other_params.append(param)        

        logger.info(
            f"Number of visual encoder params to train: {sum(param.numel() for param in visual_encoder_params)}"
        )
        logger.info(
            f"Number of other params to train: {sum(param.numel() for param in other_params)}"
        )
        logger.info(
            f"Number of total params to train: {sum(param.numel() for param in visual_encoder_params + other_params)}"
        )
        
        total_params = sum(param.numel() for param in visual_encoder_params + other_params)
        
        if total_params > 0:
            optim_cls = optim.AdamW
            optim_kwargs = dict(
                params=[
                    {"params": visual_encoder_params, "lr": encoder_lr},
                    {"params": other_params, "lr": lr},
                ],
                lr=lr,
                weight_decay=wd,
                eps=eps,
            )
            signature = inspect.signature(optim_cls.__init__)
            if "foreach" in signature.parameters:
                optim_kwargs["foreach"] = True
            else:
                try:
                    import torch.optim._multi_tensor
                except ImportError:
                    pass
                else:
                    optim_cls = torch.optim._multi_tensor.Adam

            return optim_cls(**optim_kwargs)
        else:
            logger.info("No trainable prams found -> Returning None object in place of ImageNavPPO updated !")
            return None
        
@baseline_registry.register_updater
class ImageNavDDPPO(DecentralizedDistributedMixin, ImageNavPPO):
    pass