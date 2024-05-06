import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


@torch.no_grad()
def _kaiming_uniform_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the bias of a layer according to the Kaiming uniform scheme."""

    nn.init.kaiming_uniform_(layer.weight.data[mask, ...], a=math.sqrt(5))

    if layer.bias is not None:
        if isinstance(layer, nn.Conv2d):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias[mask], -bound, bound)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(layer.bias[mask], -bound, bound)


@torch.no_grad()
def _lecun_normal_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the bias of a layer according to the Lecun normal scheme."""
    
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    # This implementation follows the jax one
    # https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L260
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    torch.nn.init.trunc_normal_(layer.weight[mask])
    layer.weight[mask] *= stddev
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias[mask])


def get_redo_masks(activations: dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = {}

    # Last activation are the q-values, which are never reset
    for name, activation in list(activations.items())[:-1]:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        if activation.ndim == 4:
            # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer
            score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks[name] = layer_mask
        
    return masks


def reset_dormant_neurons(model, redo_masks: torch.Tensor, use_lecun_init: bool):
    """Re-initializes the dormant neurons of a model."""
    # NOTE: This code only works for the Nature-DQN architecture in this repo

    layers = {}
    for name, module in model.named_modules():
        _name = name.replace('.', '_')
        layers[_name] = module
        
    _layers = []
    _masks = []
    for name, mask in redo_masks.items():
        if name in layers.keys():
            _layers.append(layers[name])
            _masks.append(mask)
        
    redo_masks = _masks[:-1]
    ingoing_layers = _layers[:-1]
    outgoing_layers = _layers[1:]

    # Sanity checks
    assert (
        len(ingoing_layers) == len(outgoing_layers) == len(redo_masks)
    ), "The number of layers and masks should match the number of masks."

    # Reset the ingoing weights
    # Here the mask size always matches the layer weight size
    for layer, mask in zip(ingoing_layers, redo_masks, strict=True):
        if torch.all(~mask):
            # No dormant neurons in this layer
            continue
        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            # The initialization scheme is the same for conv2d and linear
            # 1. Reset the ingoing weights using the initialization distribution
            if use_lecun_init:
                _lecun_normal_reinit(layer, mask)
            else:
                _kaiming_uniform_reinit(layer, mask)

    # Set the outgoing weights to 0
    for layer, next_layer, mask in zip(ingoing_layers, outgoing_layers, redo_masks, strict=True):
        if torch.all(~mask):
            # No dormant neurons in this layer
            continue
        elif isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
            # Special case: Transition from conv to linear layer
            # Reset the outgoing weights to 0 with a mask created from the conv filters
            num_repeatition = next_layer.weight.data.shape[0] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repeatition)
            next_layer.weight.data[linear_mask, :].data.fill_(0)
            if next_layer.bias is not None:
                # Need to repeat the mask for the bias
                # See https://github.com/google/dopamine/blob/485ea995655ebdf58a725dff5ec954b8847cae5f/dopamine/labs/redo/weight_recyclers.py#L642-L644
                next_layer.bias.data[linear_mask].data.fill_(0)
        elif (isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Conv2d) or
              isinstance(layer, nn.Linear) and isinstance(next_layer, nn.Linear)
            ):
            # Standard case: layer and next_layer are both conv or both linear
            # Reset the outgoing weights to 0
            next_layer.weight.data[:, mask, ...].data.fill_(0)
            if next_layer.bias is not None:
                # Need to repeat the mask for the bias
                # See https://github.com/google/dopamine/blob/485ea995655ebdf58a725dff5ec954b8847cae5f/dopamine/labs/redo/weight_recyclers.py#L642-L644
                num_repeatition = next_layer.weight.data.shape[0] // mask.shape[0]
                repeated_mask = torch.repeat_interleave(mask, num_repeatition)
                next_layer.bias.data[repeated_mask].data.fill_(0)
        else:
            continue

    return model