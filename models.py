from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(32, 64, 5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.average = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.average(x).flatten(1)

        return self.classifier(x)
