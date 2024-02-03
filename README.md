# Acoustic Scene Classification using PANN's CNN14 

This repo contains the simple code for [Acoustic Scene Classification](https://dcase.community/challenge2017/task-acoustic-scene-classification) using PANN's [CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master). 

### Prerequisite

* torch>=1.11.0
* numpy>=1.19.5
* pandas>=1.3.4
* torchaudio>=0.11.0
* scikit-learn>=0.24.2

### Getting started

1. Install the requirements: pip install -r requirements.txt
2. Download the DCASE 2017 Task1 [development](https://zenodo.org/records/400515) and [evaluation](https://zenodo.org/records/1040168) datasets into  data folder.
3. Train the CNN14 from scratch: 
   python main.py --epoch 120 --batch-size 32 --num-workers 4
4. Save and resume the model using --save and --resume arguements
5. For finetune the model, download and save pre-trained [Cnn14_mAP=0.431.pth](https://zenodo.org/records/3987831)   
6. Finetune the pre-trained CNN14 model for ASC task: python main.py --epoch 120 --batch-size 32 --num-workers 4 --pretrain

### Acknowledgement 

This code uses the models from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master).

