#  Title_Expansion_By_Deep_Learning

基于title-querys点击数据进行title侧扩充,构建相关性模型的新特征:

从高频点击的title -querys 中， 利用query分词weight和 点击次数 处理数据得到 title →  对应的所有点击query的切词及其weight.深度学习模型学习从title 到其扩展词表的概率分布

模型选取了fasttxt 和 transfomer:




> 其中transfomer 借鉴了 官方代码 https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py.

> fasttxt 是根据模型结果自己实现的

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>



# Requirement
- python 3.6+
- pytorch 1.5
- torchtext 0.6.0
- spacy 2.2.2+
- tqdm
- dill
- numpy


# Usage


### 1) Preprocess the data with torchtext
```bash
python /data/data_preprocess.py 
python preprocess_data.py -min_freq 3 ..

```

### 2) Train the model
传入自定义参数,也可以使用默认值
```bash
python train.py  -save_model trained -b 256 -warmup 1200 -epoch 400 -model_type transformer
```

### 3) predict the title
```bash
传入自定义参数,也可以使用默认值
python predict.py -output_path ***
```

---
# TODO

  - Evaluation on the generated text
  - BERT
---
# Acknowledgement
- 借鉴部分代码[OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

