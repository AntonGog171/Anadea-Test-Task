
# LIVECell Instance Segmentation task

## Overview
As a base architecture fro this task was chosen a YOLOv8, as it is light, robust and is current SOTA model for instance segmentation. It also has anchor-free architecture, and that approach was used in most literature I discovered related to this task. This solution is not perfect and can be improved, more detailed about this is told in ***Further work***. Here and everywhere below models described as "small" and "big" are *YOLOv8n* and *YOLOv8x* correspondingly. Performace of models is mostly compared with *CenterMask V-19-SlimDW* and *CenterMask V-99*, which are the most recent and best performing models on this dataset. A table with comparison of performance of these models is shown below:

### Small models comparison (both ~3M params):
| Dataset/Metric | CenterMask V-19-SlimDW | | | YOLOv8n | | |
| --- | --- | --- | --- | --- | --- | --- |
| | F1 | AP50-95 | | F1 | AP50-95 | |
| LIVECell | 0.82 | 34.7 | | 0.71 | 31.1 | |
| A172 | 0.92 | 20.5 | | 0.71 | 33.5 | |
| BT-474 | 0.82 | 27.9 | | 0.67 | 31 | |
| BV-2 | 0.81 | 46.7 | | 0.71 | 27.3 | |
| Huh7 | 0.81 | 31.3 | | 0.76 | 45.9 | |
| MCF7 | 0.88 | 25.2 | | 0.64 | 25.1 | |
| SH-SY5Y | 0.80 | 15.9 | | 0.55 | 16.7 | |
| SkBr3 | 0.90 | 61.5 | | 0.89 | 45.4 | |
| SK-OV-3 | 0.91 | 27.7 | | 0.81 | 47.3 | |

### Big models comparison (90M and 70M params):

| Dataset/Metric | CenterMask V-99 | | | YOLOv8n | | |
| --- | --- | --- | --- | --- | --- | --- |
| | F1 | AP50-95 | | F1 | AP50-95 | |
| LIVECell | 0.95 | 48.5 | | 0.77 | 35.4 | |
| A172 | 0.95 | 39.4 | | 0.76 | 39.6 | |
| BT-474 | 0.86 | 45.6 | | 0.73 | 36.3 | |
| BV-2 | 0.86 | 53.3 | | 0.75 | 29.8 | |
| Huh7 | 0.90 | 54.7 | | 0.80 | 52.2 | |
| MCF7 | 0.91 | 40.1 | | 0.72 | 31.9 | |
| SH-SY5Y | 0.84 | 27.7 | | 0.66 | 24.4 | |
| SkBr3 | 0.93 | 66.5 | | 0.91 | 48.1 | |
| SK-OV-3 | 0.94 | 54.4 | | 0.85 | 54.7 | |

## Project Structure
```
Anadea-Test-Task/
│   requirements.txt
│
├───input_examples
│
├───literature 
│
├───python_scripts
│       infere.py
│       train_notebook.ipynb
│
├───tensorboard_logs
│   ├───train_experiment_YOLO_big
│   │
│   └───train_experiment_YOLO_small
│
└───weights
        big.pt
        small.pt
```
### input_examples:
Folder with images randomly taken from dataset to show model performace
### literature:
Literature used in process.
### python_scripts:
Folder with interfere.py script to run the model and train_notebook.ipynb with notebook used for ceration of this model.
### tensorboard_logs:
Folder with tensorboard logs of training/validation of these models. Note: during training of YOLOv8x model, internall error in YOLO libarary occured, which stopped tensorboard from loggin from 15-th epoch, though progress in text format is saved along with images of results and csv logs.
### weights:
A folder with models weights.

## Setup Instructions

 Clone the Repository and install dependancies

   ```bash
   git clone https://github.com/AntonGog171/Anadea-Test-Task
   cd Anadea-Test-Task
   pip install -r requirements.txt
```

## Inference

To use model you can use next command:
   ```bash
python python_scripts/infere.py --model <small/big> --input_dir ./input_examples --output_dir ./output
```
***--model*** parameter is switching between YOLOv8n (3M params) and YOLOv8x (70M params) and can be take 2 values -  ***small*** or ***big***. This code will read all image files from given input directory and put all corresponding outputs in output folder.

## Further work
Shown models are not preforming perfectly, but have a high potential for improving, and several methods can be proposed:
#### 1) Dataset synthetic expansion
Different areas of the dataset labled as cells, can be cropped and pasted on new image, with a noisy background. This way you can at least double the dataset with new combinations of different cells on image. Mosaic agmentation was used in training, but it is not as effective as this approach and can potentially improve model performace.
#### 2) Combined models approach
Instead of using end-to-end solution, this task can be divived into 2 parts- cell type classification and further segmentation. Such approach can be more effective, as separate segmentation models for each cell type can perform better that one model for all of them.

## PS

More detailed review with model work examples will be in live presentation :)
