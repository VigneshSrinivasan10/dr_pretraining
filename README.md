This is a PyTorch implementation of the paper "To pretrain or not? A systematic analysis of the benefits of pretraining in diabetic retinopathy":
The code base as well as this README file heavuly borrows for the repository of [MoCo v2 paper](https://arxiv.org/abs/2003.04297):

### Preparation

Install PyTorch 

Download ImageNet and Eyepacs-1 datasets. 

### Pretraining

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pretraining of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

To do unsupervised pretraining of a ResNet-50 model on Eyepacs-1 in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your eyepacs1-folder with train and val folders]
```

### Downstream Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your eyepacs-folder with train and val folders]
```
This stage is always on the eyepacs-1 dataset. 

For the fully supervised setting, run the above code with `--pretrained [your checkpoint path]` removed. 


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
