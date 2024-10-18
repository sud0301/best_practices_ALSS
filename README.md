# best_practices_ALSS

## AL for Semantic Segmentation
This part is the official repository for the paper - [Best Practices in Active Learning for Semantic Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-54605-1_28#citeas)

### Datasets 
- [x] PASCAL-VOC (Augmented train set: 10582 images)
- [ ] Cityscapes
- [x] [A2D2](https://www.a2d2.audi/a2d2/en.html)
    - [x] Pool-0f
    - [x] Pool-5f
    - [x] Pool-11f
    - [x] Pool-21f
    - [ ] Pool-Aug

### Query Strategies for Semantic Segmentation
- [x] Random Sampling 
- [x] Entropy-based Sampling 
- [x] CoreSet Approach
- [x] EquAL Sampling 
- [x] Random Sampling SSL
- [x] Entropy SSL
- [x] CoreSet SSL
- [x] EquAL SSL
 
Sample script for PASCAL-VOC
```
python run.py --random-image --config ./configs/datasets/pascal_voc.yaml
python run.py --entropy-image --config ./configs/datasets/pascal_voc.yaml
python run.py --coreset --config ./configs/datasets/pascal_voc.yaml
```

Sample script for A2D2
```
python run.py --random-image --config ./configs/datasets/a2d2.yaml
```

You can download the weights for the pretrained weights (Wide-ResNet-38) for initializing the encoder backbone [here](https://drive.google.com/file/d/1h37aVCfDeUW4qmsv_3Oada5SCEJ0dMr5/view?usp=sharing)

## AL for Image Classification

### Datasets
- CIFAR-10
- CIFAR-100

### Query Strategies for Image Classification
- Random Sampling
- Entropy Sampling 
- CoreSet
- Learning Loss
- Ensemble Entropy
- Ensemble Variation Ratio (QBC)

## Citation
```
@InProceedings{ALSS_2024_GCPR,
    author    = {Mittal, Sudhanshu and Niemeijer, Joshua and Sch{\"a}fer, J{\"o}rg P. and Brox, Thomas},
    title     = {Best Practices in Active Learning for Semantic Segmentation},
    booktitle = {Proceedings of the DAGM German Conference on Pattern Recognition},
    year      = {2023},
}
```