"""
Autoencoder Network for Pacman — Next-Frame Prediction

Identical backbone to ActorCriticNetwork. The actor/critic heads are replaced
by a spatial decoder that reconstructs the next game frame from the bottleneck.

Backbone key names are intentionally identical to ActorCriticNetwork so that
a policy gradient trainer can load the backbone weights with strict=False.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with 2D convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)


class AutoencoderNetwork(nn.Module):
    """
    Convolutional autoencoder for next-frame prediction.

    Encoder: identical to ActorCriticNetwork backbone — backbone weights are
    directly compatible and can be transferred with strict=False.

    Decoder: mirrors the encoder spatially, expanding the 256-dim bottleneck
    back to the 6 base game channels (pacman, ghosts, walls, scared ghosts,
    food, capsules) at the original (H, W) resolution.

    Args:
        memory_context:      Number of past-position channels in the input.
        conv_channels:       Convolutional channels in the shared backbone.
        num_residual_blocks: Number of residual blocks.
        spatial_height:      Grid height (default 11 for mediumClassic).
        spatial_width:       Grid width  (default 20 for mediumClassic).
    """

    NUM_BASE_CHANNELS = 6  # channels to predict: pacman, ghosts, walls, scared, food, capsules

    def __init__(self, memory_context=5, conv_channels=64, num_residual_blocks=3,
                 spatial_height=11, spatial_width=20):
        super().__init__()

        self.spatial_height = spatial_height
        self.spatial_width = spatial_width

        input_channels = self.NUM_BASE_CHANNELS + memory_context

        # ── Encoder (backbone) ─────────────────────────────────────────── #
        # Key names match ActorCriticNetwork exactly for weight compatibility.
        self.conv_input = nn.Conv2d(input_channels, conv_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(conv_channels)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(conv_channels) for _ in range(num_residual_blocks)
        ])

        self.conv_output = nn.Conv2d(conv_channels, 32, 3, padding=1)
        self.bn_output = nn.BatchNorm2d(32)

        fc_input_size = 32 * spatial_height * spatial_width
        self.fc_shared = nn.Linear(fc_input_size, 256)

        # ── Decoder (expansion) ────────────────────────────────────────── #
        self.fc_decode = nn.Linear(256, 32 * spatial_height * spatial_width)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.NUM_BASE_CHANNELS, 3, padding=1),
        )

    def forward_backbone(self, x):
        """Shared feature extraction — identical to ActorCriticNetwork."""
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.residual_blocks:
            out = block(out)
        out = F.relu(self.bn_output(self.conv_output(out)))
        out = out.view(out.size(0), -1)
        return F.relu(self.fc_shared(out))

    def forward_decoder(self, bottleneck):
        """Expand bottleneck → (6, H, W) next-frame prediction."""
        out = F.relu(self.fc_decode(bottleneck))
        out = out.view(out.size(0), 32, self.spatial_height, self.spatial_width)
        return torch.sigmoid(self.decoder(out))  # (B, 6, H, W) in [0, 1]

    def forward(self, x):
        """
        Args:
            x: (B, 6+memory_context, H, W)
        Returns:
            predicted_next: (B, 6, H, W) — sigmoid probabilities per cell per channel
        """
        return self.forward_decoder(self.forward_backbone(x))

    def backbone_state_dict(self):
        """Return only the backbone weights, keyed identically to ActorCriticNetwork.

        The policy gradient trainer can load this with strict=False to warm-start
        the ActorCriticNetwork backbone from a pre-trained autoencoder.
        """
        backbone_keys = {
            'conv_input', 'bn_input', 'residual_blocks',
            'conv_output', 'bn_output', 'fc_shared'
        }
        return {
            k: v for k, v in self.state_dict().items()
            if k.split('.')[0] in backbone_keys
        }
