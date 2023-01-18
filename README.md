# GeoLS: Geodesic Label Smoothing for Image Segmentation

This repository contains implementation of GeoLS: Geodesic Label Smoothing for Image Segmentation

**Keywords:** Image Segmentation, Geodesic Distance, Label Smoothing

**TL;DR:** Geodesic distance based label smoothing for image segmentation, which captures spatial relation and image context.

## Dependencies
This code depends on the following libraries:

- GeodisTK
- python >= 3.8
- scikit-image
- SimpleITK
- torch >= 1.13.0
- torchvision >= 0.14.0

## Geodesic Maps
Install [Geodesic Distance Transform](https://github.com/taigw/GeodisTK) from :
```
pip install GeodisTK
```

Generate the normalized geodesic maps:
```python
python geodesic_maps.py --dataset=FLARE --nub_classes=5 --input_dir="./FLARE/dataset" --ouput_dir="./FLARE/geodesic_maps"
```

## GeoLS loss

Load the normalized geodesic maps along with dataset (inputs and targets) in the dataloader of any segementation network and use the geodesic maps in the loss function as below:

Example:
```python
creteria = CELossWithGeoLS(classes=5, alpha=0.1)  # For FLARE dataset having 5 classes
.
.
.
predictions = model(inputs)
loss = criterion(predictions, targets, geodesic_maps)
```

For 3D image segmentation (as in paper)

```python

class CELossWithGeoLS(torch.nn.Module):
    def __init__(self, classes=None, alpha=0.0):
        super(CELossWithGeoLS, self).__init__()
        '''
        classes: number of classes
        alpha: smoothing factor [0,1]. When alpha=0, it reduces to CE loss
        '''
        
        self.alpha = alpha
        self.classes = torch.tensor(classes)
        self.class_indx = torch.arange(self.classes).reshape(1, self.classes).cuda()


    def forward(self, predictions, targets, geodesic_maps):
        '''
        predictions and geodesic_maps of dim: [B, C, S, H, W], whereas targets of dim: [B, S, H, W]
        (Batch: B, class: C, 3D image: S x H x W)
        geodesic_maps is normalized to [0,1]
        '''
        
        with torch.no_grad():
            oh_labels = (targets[...,None] == self.class_indx).permute(0,4,1,2,3)
            gls_labels = (1 - self.alpha) * oh_labels + self.alpha * geodesic_maps
        return (- gls_labels * F.log_softmax(predictions, dim=1)).sum(dim=1).mean()
```

For 2D image segmentation:
```python
class CELossWithGeoLS_2D(torch.nn.Module):
    def __init__(self, classes=None, alpha=0.0):
        super(CELossWithGeoLS_2D, self).__init__()
        '''
        classes: number of classes
        alpha: smoothing factor [0,1]. When alpha=0, it reduces to CE loss
        '''
        
        self.alpha = alpha
        self.classes = torch.tensor(classes)
        self.class_indx = torch.arange(self.classes).reshape(1, self.classes).cuda()


    def forward(self, predictions, targets):
        '''
        predictions and geodesic_maps of dim: [B, C, H, W], whereas targets of dim: [B, H, W]
        (Batch: B, class: C, 2D image: H x W)
        geodesic_maps is normalized to [0,1]
        '''
        
        with torch.no_grad():
            oh_labels = (targets[...,None] == self.class_indx).permute(0,3,1,2)
            gls_labels = (1 - self.alpha) * oh_labels + self.alpha * geodesic_maps
        return (- gls_labels * F.log_softmax(predictions, dim=1)).sum(dim=1).mean()
```
