# LTPSF-Conv

This is a Pytorch implementation of LTPSF-Conv: "[Can Complex Networks Outperform Simple Models in Long-Term Power Time Series Forecasting?]()". 


## Features
- [x] Support both Univariate and Multivariate long-term time series forecasting.
- [x] Support visualization of weights.
- [x] Support scripts on different look-back window size.



Beside LTPSF-Conv, we provide five significant forecasting Transformers to re-implement the results in the paper.
- [x] [Transformer](https://arxiv.org/abs/1706.03762) (NeuIPS 2017)
- [x] [Informer](https://arxiv.org/abs/2012.07436) (AAAI 2021 Best paper)
- [x] [Autoformer](https://arxiv.org/abs/2106.13008) (NeuIPS 2021)
- [x] [FEDformer](https://arxiv.org/abs/2201.12740) (ICML 2022)
- [x] [PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol) (ICLR 2023)



## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n LTPSF-Conv python=3.7.13
conda activate LTPSF-Conv
pip install -r requirements.txt
```



### Data Preparation

You can obtain all the seven benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Training Example
```
- python -u run.py --is_training 1 --root_path ./dataset/ --data_path weather.csv --data custom --features M --channel 21 --itr 1 --batch_size 16 --learning_rate 0.005  --model Conv --rev --seq_len 512 --pred_len 96 --kernel_size 55 --gpu 0 
```

## Citing

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@article{xxx,
  title={Can Complex Networks Outperform Simple Models in Long-Term Power Time Series Forecasting?},
  author={},
  journal={},
  pages={},
  year={},
  publisher={}
}
```

Please remember to cite all the datasets and compared methods if you use them in your experiments.
