"""Optional residual CNN building block; no pretrained weights are distributed."""


def build_residual_cnn():
    """Build a small PyTorch model that predicts noise and subtracts it from the input."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Install PyTorch with pip install -e '.[deep]'") from exc

    class ResidualCNN(nn.Module):
        def __init__(self, width=48, depth=8):
            super().__init__()
            layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
            for _ in range(depth - 2):
                layers += [nn.Conv2d(width, width, 3, padding=1),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            layers.append(nn.Conv2d(width, 1, 3, padding=1))
            self.net = nn.Sequential(*layers)

        def forward(self, noisy):
            return noisy - self.net(noisy)

    return ResidualCNN()
