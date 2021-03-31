import torch
import torch.nn as nn
# from torchvision.models import AlexNet

def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([
        ('relu',nn.ReLU(inplace=True)),
        ('lrelu',nn.LeakyReLU())
    ])

    return nn.Sequential(
        nn.Conv2d(in_channels=in_f, out_channels=out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation],
    )

def fc_block(dropout_rate=0.5, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([
        ('relu',nn.ReLU(inplace=True)),
        ('lrelu',nn.LeakyReLU())
    ])

    return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(*args, **kwargs),
            activations[activation],
    )

class MiniAlexNet(nn.Module):
    def __init__(self) -> None:
        super(MiniAlexNet,self).__init__()
        self.features = nn.Sequential(
            conv_block(in_f=3, out_f=64, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv_block(in_f=64, out_f=192, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv_block(in_f=192, out_f=384, kernel_size=3, padding=1),
            conv_block(in_f=384, out_f=256, kernel_size=3, padding=1),
            conv_block(in_f=256, out_f=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            fc_block(dropout_rate=0.5, activation='relu', in_features=256 * 6 * 6, out_features=4096),
            fc_block(dropout_rate=0.5, activation='relu', in_features=4096, out_features=4096),
            nn.Linear(4096, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self.features(inputs)
        return x

    def forward(self, inputs):
        # See note [TorchScript super()]
        x = self.features(inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    print(MiniAlexNet())