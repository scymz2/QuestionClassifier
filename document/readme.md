# Question Classifier

`COMP61332 Text Mining Coursework 1`

`Chuhan QIU - Mingchen WAN - Mochuan ZHAN - Xinyi OUYANG - Zhangli WANG`

## Environment
`develop and test environment`

macOS 10.15.7, with 2.2 GHz 6-gen Intel Core i7 and 16 GB RAM

windows 11, with Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 2.59 GHz and RAM 16.0 GB


## Data
`training data` note: it is said to have 5500 questions but we could only find 5452 labeled questions.
https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label

`test data` https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label

## Run
```
cd src
```
To train the model, using the following command line:
```
python3 question_classifier.py --train --config [configuration_file_path]
```
After training, the trained model would be saved in the model path, then test the model using:
```
python3 question_classifier.py --test --config [configuration_file_path]
```
Please use the same configuration file path as training. Before testing the model, please make sure you have already finished training.

## NOTE
1. If you want to re-split the raw data into a new train and validation set, please delete the 'train.txt' before training, then the program will re-split the raw data.
2. Before running 'test' mode, please be sure you have already finished training.
3. The configuration files are located in `../data/configs/`, before running the program, please use the right configuration file for each model.
4. If you run the program without `--config [configuration_file_path]`, the program would read the initial configuration file `../src/config.ini`, which is the pretrain_finetuned_bow setting.
