import torch
import torch.nn as nn
import torchvision.models as tv_models
from omegaconf import DictConfig


# Maps backbone name -> (loader fn, feature dim, classifier attribute name)
_BACKBONE_REGISTRY: dict[str, tuple[callable, int, str]] = {
    "resnet18":           (tv_models.resnet18,           512,  "fc"),
    "resnet50":           (tv_models.resnet50,           2048, "fc"),
    "resnet152":          (tv_models.resnet152,          2048, "fc"),
    "densenet161":        (tv_models.densenet161,        2208, "classifier"),
    "efficientnet_b0":    (tv_models.efficientnet_b0,   1280, "classifier"),
    "efficientnet_v2_s":  (tv_models.efficientnet_v2_s, 1280, "classifier"),
    "efficientnet_v2_m":  (tv_models.efficientnet_v2_m, 1280, "classifier"),
    "mobilenet_v3_small": (tv_models.mobilenet_v3_small, 576, "classifier"),
    "convnext_tiny":      (tv_models.convnext_tiny,      768, "classifier"),
    "convnext_base":      (tv_models.convnext_base,     1024, "classifier"),
    "swin_t":             (tv_models.swin_t,              768, "head"),
    "swin_b":             (tv_models.swin_b,             1024, "head"),
    "vit_b_16":           (tv_models.vit_b_16,           768, "heads"),
}


class TransferModel(nn.Module):
    """
    Wraps any torchvision backbone for transfer learning on Food-101.

    - Replaces the original classifier head with:
        Dropout -> Linear(feat_dim, fc_hidden_size) -> ReLU -> Linear(fc_hidden_size, num_classes)
    - Optionally freezes all backbone layers, then selectively unfreezes
      from `fine_tune_from_layer` onward.
    - Exposes two param groups (backbone, head) for differential LRs.
    """

    def __init__(
        self,
        backbone: str,
        freeze_backbone: bool,
        fine_tune_from_layer: str,
        fc_hidden_size: int,
        dropout_rate: float,
        num_classes: int = 101,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone: {backbone!r}. "
                f"Available: {list(_BACKBONE_REGISTRY.keys())}"
            )

        loader, feat_dim, head_attr = _BACKBONE_REGISTRY[backbone]
        weights = "DEFAULT" if pretrained else None
        self.backbone = loader(weights=weights)
        self._backbone_name: str = backbone
        self._head_attr: str = head_attr

        # Replace classifier head
        # nn.Flatten(1) is needed for ConvNeXt whose classifier includes the flatten internally.
        # For all other backbones the input is already flat, so Flatten(1) is a no-op.
        new_head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, num_classes),
        )
        setattr(self.backbone, head_attr, new_head)

        # Freeze/unfreeze strategy
        if freeze_backbone:
            # Freeze everything except the new head
            for name, param in self.backbone.named_parameters():
                if not name.startswith(head_attr):
                    param.requires_grad = False
        elif fine_tune_from_layer != "none":
            # Freeze all layers, then unfreeze from fine_tune_from_layer onward
            unfreezing = False
            for name, param in self.backbone.named_parameters():
                if fine_tune_from_layer in name:
                    unfreezing = True
                if name.startswith(head_attr):
                    unfreezing = True
                param.requires_grad = unfreezing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def param_groups(self, lr: float, lr_multiplier_backbone: float) -> list[dict]:
        """
        Returns two optimizer param groups:
          - backbone params: lr * lr_multiplier_backbone
          - head params:     lr (full learning rate)
        Only includes params where requires_grad=True.
        """
        backbone_params = [
            p
            for name, p in self.backbone.named_parameters()
            if p.requires_grad and not name.startswith(self._head_attr)
        ]
        head_params = [
            p
            for name, p in self.backbone.named_parameters()
            if p.requires_grad and name.startswith(self._head_attr)
        ]

        return [
            {"params": backbone_params, "lr": lr * lr_multiplier_backbone},
            {"params": head_params, "lr": lr},
        ]


def build_transfer_model(cfg: DictConfig, num_classes: int = 101) -> TransferModel:
    """Build a TransferModel from a Hydra model config (fixed values, not search space)."""
    return TransferModel(
        backbone=cfg.backbone,
        freeze_backbone=cfg.freeze_backbone,
        fine_tune_from_layer=cfg.fine_tune_from_layer,
        fc_hidden_size=cfg.fc_hidden_size,
        dropout_rate=cfg.dropout_rate,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
    )
