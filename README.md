Guarding Against Deepfake Audio with One-Class Softmax
===============
This project is adapted from [AIR-ASVspoof](https://github.com/yzyouzhang/AIR-ASVspoof) and [deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features).

The project aims to explore the generalization capability of anti-spoofing systems utilizing the One-Class Softmax loss function ([Zhang et al., 2021](https://ieeexplore.ieee.org/document/9417604)) on real-world samples. Please refer to poster.pdf for more details.
* This repo: Various experiments of ResNet18.
* Whisper-softmax: Using this repo to run experiment of MFCC+Whisper+MesoNet-Softmax.
* [Whisper-ocsoftmax](https://github.com/chihyi-lin/deepfake-whisper-features): Using this repo to run experiment of MFCC+Whisper+MesoNet-OCSoftmax.

## Requirements
python==3.10
pytorch==2.3

## Data Preparation
The LFCC features are extracted with the MATLAB implementation provided by the ASVspoof 2019 organizers. 
* ASVSpoof2019: first run the `process_LA_data.m` with MATLAB, and then run `python3 reload_data.py`.
* In-the-wild: first run the `process_in_the_wild_data.m` with MATLAB, and then run `python3 reload_data.py`.
Make sure to change the directory path to the path on your machine.
## Run the training code
Before running the `train.py`, change the `path_to_features`, `path_to_protocol` according to the files' location on your machine.

E.g., LFCC-ResNet18 on ASVspoof_2019:
```
CUDA_VISIBLE_DEVICES=[idx] python train.py --dataset_version ASVspoof_2019 --
add_loss [softmax/ocsoftmax] --frontend lfcc -o ./models/[softmax/ocsoftmax] --batch_size=32 --num_epochs 20
```
Recommended configuration for ASVspoof_2021:
```
CUDA_VISIBLE_DEVICES=[idx] python train.py --add_loss ocsoftmax -o ./models_asv21_mfcc_eer_1004/ocsoftmax/ --batch_size=8 --train_amount 100000 --valid_amount 25000
```
## Run the test code with trained model
Change the `model_dir` to the location of the model you would like to test with.
```
python test.py -m ./models_asv19_lfcc_20ep/ocsoftmax -l ocsoftmax -t in_the_wild
```