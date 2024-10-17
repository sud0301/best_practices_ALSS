# best_practices_ALSS

## AL for Semantic Segmentation
This part is the official repository for the paper - Best Practices in Active Learning for Semantic Segmentation

### Datasets 
- [x] PASCAL-VOC (Augmented train set: 10582 images)
- [ ] Cityscapes
- [ ] [A2D2](https://www.a2d2.audi/a2d2/en.html)
    - [ ] Pool-0f
    - [ ] Pool-5f
    - [ ] Pool-11f
    - [ ] Pool-21f
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
 
Sample script
```
python run.py --random-image --config ./configs/datasets/pascal_voc.yaml
python run.py --entropy-image --config ./configs/datasets/pascal_voc.yaml
python run.py --coreset --config ./configs/datasets/pascal_voc.yaml
```


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