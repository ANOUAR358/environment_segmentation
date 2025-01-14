import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, layers=101):
        super(ResNetEncoder, self).__init__()
        
        # Use pre-trained ResNet model from torchvision
        if layers == 101:
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)

        # Encoder: We will use layers up to the layer3
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 convolutions with different dilation rates
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = self.conv5(x5)
        x5 = torch.nn.functional.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)

        return torch.cat([x1, x2, x3, x4, x5], dim=1)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(1024,1024)):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Variable number of output classes

        # Upsampling layer to restore original image size
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        # Upsample to the original input size
        x = self.upsample(x)
        
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, layers=101, output_channels=34):
        super(DeepLabV3Plus, self).__init__()
        
        self.encoder = ResNetEncoder(layers=layers)
        self.aspp = ASPP(2048, 256)  # Assuming the output of ResNet layer4 has 2048 channels
        self.decoder = Decoder(256 * 5, output_channels)  # Pass output_channels to decoder
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return x


import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DeepLabV3PlusModel(nn.Module):
    def __init__(self, encoder_name='efficientnet-b5', encoder_weights='imagenet', in_channels=3, num_classes=34):
        """
        Initialise le modèle DeepLabV3Plus.

        Args:
            encoder_name (str): Nom du backbone à utiliser (par défaut : 'efficientnet-b5').
            encoder_weights (str): Poids pré-entraînés pour l'encodeur (par défaut : 'imagenet').
            in_channels (int): Nombre de canaux d'entrée (par défaut : 3 pour RGB).
            num_classes (int): Nombre de classes de sortie (par défaut : 34).
        """
        super(DeepLabV3PlusModel, self).__init__()
        
        # Utilisation de la bibliothèque segmentation_models_pytorch pour créer le modèle
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    
    def forward(self, x):
        """
        Passe forward du modèle.

        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Tenseur de sortie de forme (batch_size, num_classes, height, width).
        """
        return self.model(x)