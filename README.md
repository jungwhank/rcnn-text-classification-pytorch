# RCNN for Text Classification in PyTorch

PyTorch implementation of "[Recurrent Convolutional Neural Network for Text Classification](http://zhengyima.com/my/pdfs/Textrcnn.pdf) (2015)"



## Model

![model](https://user-images.githubusercontent.com/53588015/96370598-5c3b7100-1199-11eb-9bbe-903d4ba8aeda.png)



## Requirements

```
PyTorch
sklearn
nltk
pandas
```



## Dataset

 **AG NEWS Dataset** [[Download](https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms)] : This link is from TORCHTEXT.DATASETS.

| DATASET | COUNTS  |
| :-----: | :-----: |
|  TRAIN  | 110,000 |
|  VALID  | 10,000  |
|  TEST   |  7,600  |

**Classes**

Original classes are 1, 2, 3, 4 each, but changed them into 0, 1, 2, 3.

* 0: World 

* 1: Sports

* 2: Business

* 3: Sci/Tech

  

## Training

To train,

```
python main.py --epochs 10
```

To train and want to see test set result,

```
python main.py --epochs 10 --test_set
```



## Result

For test set,

| Accuracy | Precision | Recall |   F1   |
| :------: | :-------: | :----: | :----: |
|   91.5   |  0.9154   | 0.9150 | 0.9149 |

Confusion Matrix is like below,

```
[1712   47   63   78]
[  21 1852   18    9]
[  53   18 1660  169]
[  34   24  112 1730]
```



## Reference

* Lai, S., Xu, L., Liu, K., & Zhao, J. (2015, February). Recurrent convolutional neural networks for text classification. In *Twenty-ninth AAAI conference on artificial intelligence*. [[Paper](http://zhengyima.com/my/pdfs/Textrcnn.pdf)]
