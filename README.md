# PyTorch implementation of Learning by Aligning (ICCV 2021)

This is an official PyTorch implementation of the paper "Learning by Aligning: Visible-Infrared Person Re-identification using Cross-Modal Correspondences", ICCV 2021.

For more details, visit our [project site](https://cvlab.yonsei.ac.kr/projects/LbA/) or see our [paper](https://arxiv.org/abs/2108.07422).

## Requirements
* Python 3.8<br>
* PyTorch 1.7.1<br>
* GPU memory >= 11GB<br>

## Getting started
First, clone our git repository.<br>
```
git clone https://github.com/cvlab-yonsei/LbA.git
cd LbA
```

### Docker
We provide a Dockerfile to help reproducing our results easily.<br>

### Prepare datasets
* SYSU-MM01: download from this [link](http://isee.sysu.edu.cn/project/RGBIRReID.htm).<br>
    * For SYSU-MM01, you need to preprocess the .jpg files into .npy files by running:<br>
        * `python utils/pre_preprocess_sysu.py --data_dir /path/to/SYSU-MM01`<br>
    * Modify the dataset directory below accordingly.<br>
        * L63 of `train.py`<br>
        * L54 of `test.py`<br>

## Train
* run `python train.py --method full`<br>

* **Important:**
    * Performances reported during training does <ins>**not**</ins> reflect exact performances of your model. This is due to 1) evaluation protocols of the datasets and 2) random seed configurations.<br>
    * Make sure you seperately run `test.py` to obtain correct results to be reported in your paper.<br>

## Test
* run `python test.py --method full`<br>
* The results should be around: <br>

| dataset | method | mAP | rank-1 |
| :---: | :---: | :---: | :---: |
| SYSU-MM01 | baseline | 49.54 | 50.43 |
| SYSU-MM01 | full | 54.14 | 55.41 |

### Pretrained weights
* Download [[SYSU-MM01](https://github.com/cvlab-yonsei/LbA/releases/download/v1.0/sysu_pretrained.t)]<br>
* The results should be: <br>

| dataset | method | mAP | rank-1 |
| :---: | :---: | :---: | :---: |
| SYSU-MM01 | full | 55.22 | 56.31 |

## Bibtex
```
@inproceedings{park2021learning,
  title={Learning by Aligning: Visible-Infrared Person Re-identification using Cross-Modal Correspondences},
  author={Park, Hyunjong and Lee, Sanghoon and Lee, Junghyup and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12046--12055},
  year={2021}
}
```

## Credits
Our implementation is based on [Mang Ye](https://www.comp.hkbu.edu.hk/~mangye/)'s code [here](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). 