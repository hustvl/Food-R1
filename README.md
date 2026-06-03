<div align="center">

<h1>Food-R1</h1>

<h2>A Unified Multi-Task Food Vision-Language Model with Reinforcement Learning</h2>

<p>
  Yu Zhu<sup>*</sup>, Yongkang Li<sup>*</sup>, Wenjie Zhu, Haoyi Jiang, Wenyu Liu, Wei Yang, Bin Li, <a href="https://xwcv.github.io/">Xinggang Wang</a><sup>†</sup>
</p>

<p>
  Huazhong University of Science and Technology
</p>

<p>
  <sup>*</sup>Equal contribution, <sup>†</sup>Corresponding author: 
  <a href="mailto:xgwang@hust.edu.cn">xgwang@hust.edu.cn</a>
</p>

<p>
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/collections/zy12123/food-r1">
    <img src="https://img.shields.io/badge/HuggingFace-Model%20%26%20Data-yellow.svg" alt="HuggingFace">
  </a>
</p>

</div>


## Installation

For SFT, create the SFT environment:

```bash
conda env create -f environment.yml
conda activate foodr1-sft
```

For GRPO, create the GRPO environment:

```bash
conda env create -f environment_grpo.yml
conda activate foodr1-grpo
```

Food-R1 training is based on `ms-swift`. 

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```


## Datasets

Food-R1 is trained and evaluated on multiple public food-related datasets. Please download the datasets from their official sources and follow their respective licenses and access policies.

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [VIREO Food-172](https://fvl.fudan.edu.cn/dataset/vireofood172/list.htm), [Recipe1M](https://im2recipe.csail.mit.edu/), [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k), [FoodDialogues](https://huggingface.co/datasets/Yueha0/FoodDialogues), [MM-Food-100K](https://huggingface.co/datasets/Codatta/MM-Food-100K)

Note: For Nutrition5k, you only need to download the extracted Nutrition5k images provided in [FoodDialogues](https://huggingface.co/datasets/Yueha0/FoodDialogues) instead of the original Nutrition5k image data.



## Training

First configure the local paths:

```bash
cp configs/foodr1.env.example local.env
```

Edit `local.env` with your `ms-swift` checkout, model checkpoint, dataset files, image roots, and output directories.

Run SFT:

```bash
FOODR1_ENV=local.env bash scripts/train_sft.sh
```

Run GRPO on CalorieBench-80K:

```bash
FOODR1_ENV=local.env bash scripts/train_grpo.sh
```

## Evaluation

The public release includes CalorieBench-80K evaluation.

Run inference:

```bash
python eval/infer_caloriebench80k.py \
  --model /path/to/foodr1/checkpoint \
  --data /path/to/caloriebench80k_val.json \
  --image_prefix /path/to/caloriebench80k/images \
  --output outputs/pred/caloriebench80k_val \
  --num_processes 8 \
  --batch_size 8
```

Compute metrics:

```bash
python eval/metrics_caloriebench80k.py --input outputs/pred/caloriebench80k_val.json
```


## Acknowledgement

We thank the contributors of [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), and the public food datasets used in this work. We also acknowledge [GPT-4](https://openai.com/index/gpt-4-research/)  for supporting the construction and annotation of CalorieBench-80K.    


## License

This project is released under the Apache License 2.0.