#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as torch_models


# class ResNetWithClassifier(nn.Module):
#     def __init__(self, input_shape, num_classes, pretrained=True, requires_grad=True):
#         super(ResNetWithClassifier, self).__init__()

#         # Load the pre-trained ResNet model
#         if pretrained:
#             resnet = torch_models.resnet18(weights=torch_models.ResNet18_Weights.DEFAULT)
#         else:
#             resnet = torch_models.resnet18(weights=None)

#         # # Replace inplace ReLU with regular ReLU
#         # for module in resnet.modules():
#         #     if isinstance(module, nn.ReLU):
#         #         module.inplace = False

#         # Remove the fully connected layer of ResNet
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)

#         # Freeze the pre-trained layers
#         if not requires_grad:
#             for param in self.resnet.parameters():
#                 param.requires_grad=False
#         else: # to train the hidden layers
#             for param in self.resnet.parameters():
#                 param.requires_grad=True

#         self.dropout1 = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(self._calculate_conv_output_size(input_shape), num_classes)

#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout1(x)
#         x = self.fc1(x)
#         return x

#     def load_weights(self, path):
#         self.load_state_dict(torch.load(path))

#     def _calculate_conv_output_size(self, input_shape):
#         x = torch.randn(1, *input_shape)
#         out = self.resnet(x)
#         out_size = out.size(1) * out.size(2) * out.size(3)
#         return out_size



def model(pretrained, requires_grad, num_classes):
    model = torch_models.resnet50(weights=torch_models.ResNet50_Weights.DEFAULT)
    
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad=False
            
     # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad=True
            
    # make the classification layer learnable, i have 15 classes
    model.fc = nn.Sequential(
        nn.Linear(2048,num_classes))
    return model