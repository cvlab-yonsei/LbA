# PyTorch implementation of Learning by Aligning (ICCV 2021)

This is an official PyTorch implementation of the paper "Learning by Aligning: Visible-Infrared Person Re-identification using Cross-Modal Correspondences", ICCV 2021.

For more details, visit our [project site](https://cvlab.yonsei.ac.kr/projects/LbA/) or see our [paper]().

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
<br>

### Docker
We will provide a Dockerfile to help reproducing our work easily soon.<br>
For now, you can use `docker pull sanghslee/ps:1.7.1-cuda11.0-cudnn8-runtime`<br>

### Prepare datasets
* SYSU-MM01: download via this [link]().<br>
    * For SYSU-MM01, you need to preprocess the .jpg files into .npy files by running:<br> 
        * `python utils/pre_preprocess_sysu.py --data_dir /path/to/SYSU-MM01`<br>
    * Modify the dataset directory below accordingly.<br>
        * L63 of `train.py`<br>
        * L54 of `test.py`<br>

## Train
* To train our full model, run `python train.py --method full`<br>

> <span style="color:red">**Important:**</span><br>
> * Performances reported during training does <u>**not**</u> reflect exact performances of your model. This is due to 1) evaluation protocols of the datasets and 2) random seed configurations.<br>
> * Make sure you seperately run `test.py` to obtain correct result to be reported in your paper.<br>

## Test
* To test our full model, run `python test.py --method full`<br>
* The results should be around: <br>

| dataset | method | mAP | rank-1 |
| :---: | :---: | :---: | :---: |
| SYSU-MM01 | baseline | 49.54 | 50.43 |
| SYSU-MM01 | full | 54.14 | 55.41 |


### Pretrained models
* We will provide weights of our best model soon.<br>

## Bibtex
```
@inproceedings
```
<br>

## Todo
- [ ] provide dockerfile<br>
- [ ] provide pretrained weights<br>
- [ ] update paper link<br>
- [ ] update bibtex<br>


## Credits
Our implementation is based on [Mang Ye](https://www.comp.hkbu.edu.hk/~mangye/)'s code [here](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). 