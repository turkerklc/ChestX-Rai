import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class HybridDenseNet121(nn.Module):
      """
      DenseNet'i ilgili Stanford makalesi yüzünden ResNet'e tercih ettim. 
      O makaleye ek olarak hasta meta datalarını (yaş ve cinsiyet) da modele ekledim
      """

      def __init__(self, num_classes = 14, pretrained = True):
            super(HybridDenseNet121, self).__init__()
            print(f"DenseNet yükleniyor... (Pretrained: {pretrained})")
            self.densenet = models.densenet121(weights='DEFAULT' if pretrained else None)

            self.features = self.densenet.features

            self.num_image_features = 1024 #Modelde son katmandan çıkan özellik sayısı sabit.

            self.num_meta_data = 2 #Meta data olarak aldığımız özellik sayısı 2. (yaş ve cinsiyet)

            # yeni classifier katmanı

            self.classifier = nn.Sequential(
                  nn.Linear(self.num_image_features + self.num_meta_data, 512),
                  nn.ReLU(),
                  nn.Dropout(0.2),
                  nn.Linear(512, num_classes)
            )
      
      def forward(self, image, metadata):
            f = self.features(image)
            f = F.relu(f,inplace=True)
            f = F.adaptive_avg_pool2d(f ,(1,1))
            f = torch.flatten(f,1)

            #meta data ile görüntü vektörünü yan yana koyuyoruz burada
            combined = torch.cat((f,metadata), dim = 1)

            output = self.classifier(combined)
            return output 