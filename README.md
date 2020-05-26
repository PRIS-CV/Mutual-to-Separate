# Mind the Gap: Enlarging the Domain Gap in Open Set Domain Adaptation

Code release for Mind the Gap: Enlarging the Domain Gap in Open Set Domain Adaptation [ArXiv](https://arxiv.org/abs/2003.03787 "ArXiv")

## Dataset
### OFFICE-Home

## Requirements

- python 3.6
- PyTorch 1.1.0
- torchvision 0.3.0
- Tensorflow 1.9.0 
- Tensorlayer 1.11
- Tensorboard 
- tensorpack

## GPU Version

- 1080ti

## Training

- Download datasets
- Train: `python Office_Home.py  "Art"   "Clipart" "0" "A_C" 0.2 0.2`
- Description : PyTorch Open-set OFFICE-HOME Training with ResNet50 (PRE-TRAINED WITH IMAGENET).

## Citation
please cite:
```
@misc{chang2020mind,
    title={Mind the Gap: Enlarging the Domain Gap in Open Set Domain Adaptation},
    author={Dongliang Chang and Aneeshan Sain and Zhanyu Ma and Yi-Zhe Song and Jun Guo},
    year={2020},
    eprint={2003.03787},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Reference codes
**https://github.com/thuml/easydl**

## Contact
- changdongliang@bupt.edu.cn
- mazhanyu@bupt.edu.cn
