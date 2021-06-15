## save & load model
import torch
import torch.onnx as onnx
import torchvision.models as models

model = models.vgg16(pretrained=True)

## 1)save weights
torch.save(model.state_dict(), 'model_weights.pth')
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

## 2)save the structure
torch.save(model, 'model.pth')
model = torch.load('model.pth')

## 3)onnx  see: https://github.com/onnx/tutorials
