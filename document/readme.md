# Question Classifier

`COMP61332 Text Mining Coursework 1`

`Chuhan QIU - Mingchen WAN - Mochuan ZHAN - Xinyi OUYANG - Zhangli WANG`

## Environment
`develop environment`macOS 10.15.7, python 3.8, pytorch stable(1.10.2) / windows 11, python 3.7, pytorch stable(1.10.2)

`test environment`Manchester computer science virtual machine Ubuntu (64.bit) CSImage 2122 v15 PGT 4GB RAM CPU 3

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

## Note
Normally, when training the model, the test result would not 