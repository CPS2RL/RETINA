"""
loss.py
-------
Multi-exit joint loss function.

The paper trains all 3 classifier heads simultaneously.
Total loss = weighted sum of cross-entropy losses from each stage head.

  L_total = w1*CE(head1) + w2*CE(head2) + w3*CE(head3)

Default weights [0.3, 0.3, 0.4] put slightly more emphasis on the
final stage (which has the most information), while still teaching
the earlier stages to produce useful early predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiExitLoss(nn.Module):
    """
    Weighted sum of cross-entropy losses across all exit heads.

    Args:
        stage_weights : list of floats, one weight per stage.
                        Must sum to 1 (or any positive value — they are
                        used as-is in the weighted sum).
                        Default [0.3, 0.3, 0.4].
        label_smoothing : float, label smoothing factor (helps calibration).
                          Default 0.1.
    """

    def __init__(self, stage_weights=None, label_smoothing=0.1,
                class_weights=None):
        super().__init__()
        if stage_weights is None:
            stage_weights = [0.3, 0.3, 0.4]
        assert len(stage_weights) == 3, "Provide exactly 3 stage weights."

        self.stage_weights   = stage_weights
        self.label_smoothing = label_smoothing
        # Per-class weights (inverse frequency), applied at every stage's
        # cross-entropy. Needed for imbalanced datasets like ROAD, where
        # Ped outnumbers Mobike by ~60x. None -> unweighted (CIFAR-10 case).
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None else torch.tensor([])
        )

    def forward(self, stage_outputs, targets):
        """
        Args:
            stage_outputs : list of stage dicts from model.forward()
                            Each dict has key 'logits' : (B, num_classes)
            targets       : (B,) ground-truth class indices

        Returns:
            total_loss      : scalar weighted sum
            per_stage_losses: list of scalar losses (for logging)
        """
        per_stage_losses = []

        weight = self.class_weights if self.class_weights.numel() > 0 else None

        for out in stage_outputs:
            logits = out['logits']                  # (B, num_classes)
            loss   = F.cross_entropy(
                logits, targets,
                label_smoothing=self.label_smoothing,
                weight=weight,
            )
            per_stage_losses.append(loss)

        # Weighted sum — pad weights if fewer than 3 stages executed
        total_loss = sum(
            self.stage_weights[i] * per_stage_losses[i]
            for i in range(len(per_stage_losses))
        )

        return total_loss, per_stage_losses


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    B, C = 8, 10
    fake_outputs = [
        {'logits': torch.randn(B, C), 'stage': s}
        for s in [1, 2, 3]
    ]
    targets    = torch.randint(0, C, (B,))
    criterion  = MultiExitLoss()
    total, per = criterion(fake_outputs, targets)
    print(f"Total loss : {total.item():.4f}")
    for i, l in enumerate(per):
        print(f"  Stage {i+1} loss : {l.item():.4f}")
