# Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model

**Abstract:** In ophthalmic surgery, developing an AI system capable of interpreting surgical videos and predicting subsequent operations requires numerous ophthalmic surgical videos with high-quality annotations, which are difficult to collect due to privacy concerns and labor consumption. Text-guided video generation (T2V) emerges as a promising solution to overcome this issue by generating ophthalmic surgical videos based on surgeon instructions. In this paper, we present Ophora, a pioneering model that can generate ophthalmic surgical videos following natural language instructions. To construct Ophora, we first propose a Comprehensive Data Curation pipeline to convert narrative ophthalmic surgical videos into a large-scale, high-quality dataset comprising over 160K video-instruction pairs, Ophora-160K. Then, we propose a Progressive Video-Instruction Tuning scheme to transfer rich spatial-temporal knowledge from a T2V model pre-trained on natural video-text datasets for privacy-preserved ophthalmic surgical video generation based on Ophora-160K. Experiments on video quality evaluation via quantitative analysis and ophthalmologist feedback demonstrate that Ophora can generate realistic and reliable ophthalmic surgical videos based on surgeon instructions. We also validate the capability of Ophora for empowering downstream task of ophthalmic surgical workflow understanding.


## Introduction

This repository is for our work submitted to MICCAI25, titled "Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model".

We have released the training and inference codes of Ophora. The model checkpoint will be released after the review process.

![Framework](./ophora.png)

## To prepare dataset
```bash
bash prepare_dataset.sh
```

## Train
```bash
bash TPT.sh
bash P2FT.sh
```

## Inference
```bash
bash sample.sh
```


