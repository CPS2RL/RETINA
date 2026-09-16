import torch
import torch.nn as nn
import torch.nn.functional as F

# building blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    "ResNet bottleneck block "
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,                  1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,    planes,                  3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,    planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


#  Classifier head attached after each stage
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x          = self.gap(x).flatten(1)
        logits     = self.fc(x)
        probs      = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)
        return logits, probs, confidence, predicted


#  Multi-Exit ResNet  —  CIFAR-10
class MultiExitResNetCIFAR(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Shared stem
        self.conv0 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn0   = nn.BatchNorm2d(16)

        # Stage 1: 5 BasicBlocks, 16ch, stride=1
        self.stage1 = self._make_stage(BasicBlock, 16, 16, num_blocks=5, stride=1)
        self.head1  = ClassifierHead(16, num_classes)

        # Stage 2: 5 BasicBlocks, 32ch, stride=2
        self.stage2 = self._make_stage(BasicBlock, 16, 32, num_blocks=5, stride=2)
        self.head2  = ClassifierHead(32, num_classes)

        # Stage 3: 5 BasicBlocks, 64ch, stride=2
        self.stage3 = self._make_stage(BasicBlock, 32, 64, num_blocks=5, stride=2)
        self.head3  = ClassifierHead(64, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(block, in_planes, planes, num_blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, exit_after=None):
        outputs = []
        x = F.relu(self.bn0(self.conv0(x)))

        x = self.stage1(x)
        logits, probs, conf, pred = self.head1(x)
        outputs.append({'logits': logits, 'probs': probs,
                        'confidence': conf, 'predicted': pred, 'stage': 1})
        if exit_after == 1:
            return outputs

        x = self.stage2(x)
        logits, probs, conf, pred = self.head2(x)
        outputs.append({'logits': logits, 'probs': probs,
                        'confidence': conf, 'predicted': pred, 'stage': 2})
        if exit_after == 2:
            return outputs

        x = self.stage3(x)
        logits, probs, conf, pred = self.head3(x)
        outputs.append({'logits': logits, 'probs': probs,
                        'confidence': conf, 'predicted': pred, 'stage': 3})
        return outputs


#  Multi-Exit ResNet  —  ImageNet
class MultiExitResNetImageNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage1  = self._make_layer(Bottleneck,  64,  64,  blocks=3, stride=1)
        self.head1   = ClassifierHead(256,  num_classes)

        self.stage2a = self._make_layer(Bottleneck, 256, 128,  blocks=4, stride=2)
        self.stage2b = self._make_layer(Bottleneck, 512, 256,  blocks=6, stride=2)
        self.head2   = ClassifierHead(1024, num_classes)

        self.stage3  = self._make_layer(Bottleneck, 1024, 512, blocks=3, stride=2)
        self.head3   = ClassifierHead(2048, num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(block, in_planes, planes, blocks, stride):
        layers = [block(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, exit_after=None):
        outputs = []
        x = self.stem(x)

        x = self.stage1(x)
        logits, probs, conf, pred = self.head1(x)
        outputs.append({'logits': logits, 'probs': probs,
                        'confidence': conf, 'predicted': pred, 'stage': 1})
        if exit_after == 1:
            return outputs

        x = self.stage2a(x)
        x = self.stage2b(x)
        logits, probs, conf, pred = self.head2(x)
        outputs.append({'logits': logits, 'probs': probs,
                        'confidence': conf, 'predicted': pred, 'stage': 2})
        if exit_after == 2:
            return outputs

        x = self.stage3(x)
        logits, probs, conf, pred = self.head3(x)
        outputs.append({'logits': logits, 'probs': probs,
                        'confidence': conf, 'predicted': pred, 'stage': 3})
        return outputs


#  Factory
def build_model(dataset: str) -> nn.Module:
    if dataset == 'cifar10':
        return MultiExitResNetCIFAR(num_classes=10)
    elif dataset == 'imagenet':
        return MultiExitResNetImageNet(num_classes=1000)
    elif dataset == 'road':
        return MultiExitResNetCIFAR(num_classes=6)
    elif dataset == 'road_behavior':
        return MultiExitResNetCIFAR(num_classes=16)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose 'cifar10', 'imagenet', "
            f"'road', or 'road_behavior'.")


#  sanity check
if __name__ == '__main__':
    print("=== CIFAR-10 model ===")
    model = build_model('cifar10')
    x     = torch.randn(4, 3, 32, 32)
    outs  = model(x)
    for o in outs:
        print(f"  Stage {o['stage']}  logits={o['logits'].shape}  "
              f"avg_conf={o['confidence'].mean().item():.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("\n=== ImageNet model ===")
    model = build_model('imagenet')
    x     = torch.randn(2, 3, 224, 224)
    outs  = model(x)
    for o in outs:
        print(f"  Stage {o['stage']}  logits={o['logits'].shape}  "
              f"avg_conf={o['confidence'].mean().item():.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
