import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")


class VGG16Features(nn.Module):
    def __init__(self):
        super(VGG16Features, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.features(x)


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        x = self.fc(h_n)
        return x

class TransferModel(nn.Module):
    def __init__(self, num_classes):
        super(TransferModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TriFusion(nn.Module):
    def __init__(self, vgg_features, bigru, transfer_model, dropout_rate=0.5):
        super(TriFusion, self).__init__()
        self.vgg_features = vgg_features
        self.bigru = bigru
        self.transfer_model = transfer_model
        self.fc1 = nn.Linear(512 + 101 + 101, 303)
        self.bn = nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(303, 101)

    def forward(self, x):
        vgg_output = self.vgg_features(x)
        vgg_output = vgg_output.view(vgg_output.size(0), -1, 512)
        _, h_n = self.bigru.gru(vgg_output)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        bigru_output = self.bigru.fc(h_n)

        transfer_output = self.transfer_model(x)

        concatenated_output = torch.cat((vgg_output.mean(dim=1), bigru_output, transfer_output), dim=1)
        concatenated_output = self.fc1(concatenated_output)
        concatenated_output = self.bn(concatenated_output)
        concatenated_output = self.relu(concatenated_output)
        concatenated_output = self.dropout(concatenated_output)
        concatenated_output = self.fc2(concatenated_output)

        return concatenated_output



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_features = VGG16Features().to(device)
bigru = BiGRU(input_size=512, hidden_size=256, num_classes=101).to(device)
transfer_model = TransferModel(num_classes=101).to(device)

hybrid_model = TriFusion(vgg_features, bigru, transfer_model).to(device)

summary(hybrid_model, (3, 224, 224))

