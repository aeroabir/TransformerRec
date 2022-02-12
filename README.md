# TransformerRec: Sequential Recommendation Using Transformers

Most of the transformer based methods use negative sampling and do not predict the next product directly. 
This repository provides multiple encoder-decoder models to directly predict the next item in the sequence
and sequence of items in the next session/basket. 

## Datasets

1. Dunhumby dataset: this dataset contains interactions of 2500 households with average 111 baskets per households
 and 9 items per basket. There are total 92,339 items. 

2. H&M dataset: this is from Kaggle competition with 1,362,281 users, 105,542 items and 31,788,324 interactions. The
 objective is to predict the next purchase items for all the users within next 7 days after the training period. 

## Algorithms

1. Transformer based Encoder to predict the next item

## Usage

 1. python main.py --dataset hnm_small --train_dir train --model_name t-encoder --batch_size 512


