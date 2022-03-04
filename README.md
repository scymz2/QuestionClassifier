# QuestionClassifier

## contributors (sort by surname)
The university of manchester Postgards - March 2022

Coursework 1 for text mining comp61332

Xinyi Ouyang - Chuhan Qiu - Mingchen Wan - Zhangli Wang - Mochuan Zhan

## pipeline
```
word tokenization->word embedding->sentence representation (BOW, BiLSTM)->training classifier (NN)
```
![Pipeline](https://github.com/scymz2/QuestionClassifier/edit/master/document/img/pipeline.jpg)
## folder structure
```
.
├── README.md
├── data
│   ├── dev.txt
│   ├── glove.small.txt
│   ├── raw_data.txt
│   ├── stopword.txt
│   ├── train.txt
│   ├── test.txt
│   └── vocabulary.txt
├── document
│   ├── README.md
│   ├── document.md
│   └── document.pdf
├── src
│   ├── sentence_rep
│   │   ├── __init__.py
│   │   ├── biLSTM.py
│   │   └── bow.py
│   ├── utility
│   │   ├── __init__.py
│   │   ├── file_loader.py
│   │   └── pre_train.py
│   ├── __init__.py
│   ├── config.ini
│   ├── model.py
│   └──  question_classifier.py  
└──

```

## environment
`System environment for testing` Manchester computer science virtual machine Ubuntu (64.bit) CSImage 2122 v15 PGT
4GB RAM CPU 3

`hardware environment ` Huawei Matebook14 2019 windows 11, with 8-gen Core i5 CPU and 8GB RAM

`Training dataset` [5500-labeled questions](https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label)

`Testing dataset` [TREC 10](https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label)

## run

