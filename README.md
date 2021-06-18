# FlyExpress 
This is the course project in SJTU EI314-1 (上海交通大学工程实践与科技创新3). In this project, we realize a novel model to achieve good performance over flyexpress dataset

|                | AUC     | Macro F1 |         |
|----------------|---------|----------|---------|
| validation set |  0.9590 | 0.6477   | 0.6593  |
|    test set    | 0.95561 | 0.62405  | 0.61615 |

More detailed information can refer to the report [Fast regulated network over variable sets of features
with loss annealing](https://github.com/daxixi/Flyexpress-pytorch/blob/main/report.pdf)

## Pretrained Model
pretrianed model are provided, put them under folder premodel
## Dataset
Please download dataset from [Jbox](https://jbox.sjtu.edu.cn/l/6Flzxt), then put the file in this path tree

```
-data
    -train
        -train
            -0002907767_s.bmp
            -0008119820_s.bmp
            -...
        -final_result_sample.csv
        -label_dict.csv
        -train.csv
    -test
        -test
            -00001695937_s.bmp
            -...
        -test_without_label
```
## Train
### Step1: Warmup
A. You can use pretrained model ```step1.pt``` from [GoogleDrive](https://drive.google.com/drive/folders/1LK-tqzuu0vgN7YcEWMmxBkK6vyRiTBdj?usp=sharing) or [Jbox](https://jbox.sjtu.edu.cn/l/lFFs1M)

B. You can also train yourself. Directly run the command
```
python3 main.py
```
The parameter is aready set to default, or you can also refer to the report
### Step2: CNN Extraction
Put the trained model in step1 in premodel

A. You can use pretrained model ```step2.pt``` from [GoogleDrive](https://drive.google.com/drive/folders/1LK-tqzuu0vgN7YcEWMmxBkK6vyRiTBdj?usp=sharing) or [Jbox](https://jbox.sjtu.edu.cn/l/lFFs1M)

B. You can also train yourself. Directly run the command
```
python3 main.py
```
The parameter is aready set to default, or you can also refer to the report

### Step3: RNN
Put the trained model in step2 in premodel
#### Generate features
A. You can use generated features ```feature.pickle``` from [GoogleDrive](https://drive.google.com/drive/folders/1LK-tqzuu0vgN7YcEWMmxBkK6vyRiTBdj?usp=sharing) or [Jbox](https://jbox.sjtu.edu.cn/l/lFFs1M)

B You can also generate yourself by directly running the command
```
python3 generate_feature.py
```
#### Train RNN
A. You can use pretrained model ```step3.pt``` from [GoogleDrive](https://drive.google.com/drive/folders/1LK-tqzuu0vgN7YcEWMmxBkK6vyRiTBdj?usp=sharing) or [Jbox](https://jbox.sjtu.edu.cn/l/lFFs1M)

B. You can also train yourself. Directly run the command
```
python3 main.py
```
The parameter is aready set to default, or you can also refer to the report

## Test
Put the trained model in step3 in premodel

To get the test result, you may need generate test features by 
```
python3 generate_feature_test.py
```
or directly use generated features ```feature_test.pickle``` from [GoogleDrive](https://drive.google.com/drive/folders/1LK-tqzuu0vgN7YcEWMmxBkK6vyRiTBdj?usp=sharing) or [Jbox](https://jbox.sjtu.edu.cn/l/lFFs1M) 

The trained model is ```step3.pt``` then directly run
```
python3 submit.py 
```
## Submission
submission文件夹下的两个文件是我交到kaggle上用的

## Acknowledgements
We give our sincerest appreciation to Prof.Yang and Dr. Tu
for giving us this chance to contribute our efforts in the work
of drosophila embryos multilabel classification. We also thank
them for their generous help
## Contact
Jimmyyao18@sjtu.edu.cn