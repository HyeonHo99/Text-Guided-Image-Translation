# Text-Guided-Image-Translation
<ul>
  <li>Image-to-Image Translation based on Cycle-consistency loss and CLIP directional loss</li>
  <li>Graduation Thesis Project - Jeong Hyeonho Sungkyunkwan University, College of Computing</li>
</ul>

## Quick Start
### Train from scratch

```consle
  $python train.py --mode fl-clip --data horse2zebra --epoch 200
```
<ul>
  <li><b>mode</b> : loss mode (options: original / fl / clip / fl-clip)</li>
  <li><b>data</b> : name of the dataset (ex: horse2zebra, apple2orange, winter2summer, cat2dog..) (For exact details, refer to 'dataset' folder)</li>
  <li><b>epoch</b> : the number of epochs to train (default: 200)</li>
  <li><ul>
    <b>other arguments</b>
    <li><b>lr_decy</b> : True => linear decay of learning rate enabled (starting from 100 epochs) (if True, after 200 epoch, learning rate reaches 0)</li>
    <li><b>gpu</b> : which n-th gpu to use (ex. 0,1,2)
    <li><b>batch_size</b> : the number of batch size
    </ul></li>
</ul>
  
### Inference using pretrained weights

```consle
  $python inference.py
```

## Abstract

## Preliminaries

## Methods

## Results - Qualitative Analysis


## Results - Quantitative Analysis

