"""U-Net with separate centroid head for joint seg + counting."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetWithCentroids(nn.Module):
    """U-Net with a shared decoder but separate output heads.

    The segmentation head (from smp) produces 2-class logits.
    A separate centroid head produces a 1-channel heatmap from the same
    decoder features, giving it its own learnable parameters to transform
    shared features into point predictions.

    Output: [B, 3, H, W] — channels 0-1 seg logits, channel 2 centroid logits.
    """

    def __init__(
        self, encoder_name: str, encoder_weights: str = "imagenet", in_channels: int = 3
    ):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=2,  # seg only: bg + fg
        )
        # Decoder output channels (input to segmentation_head)
        decoder_out_ch = self.unet.segmentation_head[0].in_channels

        # Deeper centroid head with dilated convolutions for larger receptive
        # field — lets the head "look across" objects to find centers rather
        # than just reflecting edge features from the decoder.
        self.centroid_head = nn.Sequential(
            nn.Conv2d(decoder_out_ch, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # CenterNet-style bias init: sigmoid(-4) ≈ 0.018, so initial
        # predictions are near-zero ("assume background") instead of 0.5.
        nn.init.constant_(self.centroid_head[-1].bias, -4.0)

    def forward(self, x):
        features = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        seg = self.unet.segmentation_head(decoder_out)
        centroid = self.centroid_head(decoder_out)
        return torch.cat([seg, centroid], dim=1)  # [B, 3, H, W]
