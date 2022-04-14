# Ono-Project

## Summary 
- This reconstruction-based approach is based on predicting the future frame and comparing it with the ground truth frame based on a score called PSNR(Peak Signal to noise ratio) helps in determining whether there exist anomaly in the frame or not
- More the PSNR value, highly probable the frame is normal and less value of PSNR signifies the frame being anomalous
- Also, further in this readme there is a mention of threshold which helps in labelling the frame based on PSNR 
- Below is the architecture of the model
<img src="https://github.com/ghatoleyash/Ono-Project/blob/main/Images/Architecture.png" width=80% height=50%>

## Software to install
- Install Anaconda click [here](https://docs.anaconda.com/anaconda/install/index.html)
- Creating the virtual environment
```
conda create --name myenv
```
- activating the virtual environment
```
conda activate myenv
```
- Install python 3.7.12
```
conda install -c anaconda python=3.7.12
numpy==1.14.1
scipy==1.0.0
matplotlib==2.1.2
tensorflow-gpu==1.4.1
tensorflow==1.4.1
Pillow==5.0.0
pypng==0.0.18
scikit_learn==0.19.1
opencv-python==3.2.0.6
```
- Install NumPy, scipy, matplotlib, tensorflow-gpu, tensorflow, Pillow, pypng, scikit-learn, opencv-python
```
conda install -c library=specified_version
```

- CUDA (10.0) install click [here](https://developer.nvidia.com/cuda-10.0-download-archive)
- CUDNN (7.5) install click [here](https://developer.nvidia.com/cudnn)

## Running the current implementation
- Download the trained models (There are the pretrained FlowNet and the trained models of the papers, such as ped1, ped2 and avenue). Please manually download pretrained models from [pretrains.tar.gz, avenue, ped1, ped2, flownet](http://101.32.75.151:8181/dataset/) and tar -xvf pretrains.tar.gz, and move pretrains into Codes/checkpoints folder.
- add the video to the following path *'../Data/ped1/testing/'*
- Run the following command (runs the code)
```
python generatePred.py
```
- Asks for *Enter the video name with extension: (example.mp4)*
- Once, the predictions are completed code prompts *"Press 1 Comparison with GroundTruth else 0"*
- If pressed *1*, an excel find named *PSNRS.csv* would be created within */Codes* folder add the frame by frame label in *Ground Truth* column there
  - 0: Anomaly
  - 1: Normal
  - As the annotations are completed press *ctrl+c* to interrupt
  - End video would be generated here named *../Data/ped1/testing/frames/ResultVideo/exampleResult.mp4*
- Else, only model prediction video is generated here named *../Data/ped1/testing/frames/ResultVideo/exampleResult.mp4*



## Guide over Current Implementation on Cow Data
**[Data Generation](https://github.com/ghatoleyash/Ono-Project/blob/main/Codes/Data%20Generation/DataGeneration.md)**

**Illustration of Training Process**
<img src="https://github.com/ghatoleyash/Ono-Project/blob/main/Images/Training_Process.png" width=80% height=50%>

**Testset performance**
-  Model trained on 46,713 normal frames
-  While the model is tested on 1,250 frames out which 909 frames are anomalous and the remaining 341 are normal frames
-  Below table shows the confusion matrix on the test set
<img src="https://github.com/ghatoleyash/Ono-Project/blob/main/Images/Test_Results.png" width=60% height=50%>

- Performance metrics: Accuracy:- 77%, Precision:- 77.56%, Recall:- 83.38%, F1:- 80.36%
- Threshold: 0.73 (based on experimental result) to distinguish the between anomalous and normal frame, if the score of the frame is below threshold value then it is tagged as anomalous frame else given as normal frame




## Running From Scratch
### Dataset download for testing
cd into Data folder of project and run the shell scripts [ped1.sh, ped2.sh, avenue.sh, shanghaitech.sh](http://101.32.75.151:8181/dataset/) under the Data folder. Please manually download all datasets from ped1.tar.gz, ped2.tar.gz, avenue.tar.gz and shanghaitech.tar.gz and tar each tar.gz file, and move them in to Data folder.

### Downloading the pre-trained weights
- Download the trained models (There are the pretrained FlowNet and the trained models of the papers, such as ped1, ped2 and avenue). Please manually download pretrained models from [pretrains.tar.gz, avenue, ped1, ped2, flownet](http://101.32.75.151:8181/dataset/) and tar -xvf pretrains.tar.gz, and move pretrains into Codes/checkpoints folder.

- Running the sript (as ped2 and avenue datasets for examples) and cd into Codes folder at first. Below example is for ped1 dataset
```
python inference.py  --dataset  ped1    \
                    --test_folder  ../Data/ped1/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    checkpoints/pretrains/ped1
```

### Training from scratch 
- Download the pretrained FlowNet at first and see above mentioned step 3.1
- Set hyper-parameters The default hyper-parameters, such as $\lambda_{init}$, $\lambda_{gd}$, $\lambda_{op}$, $\lambda_{adv}$ and the learning rate of G, as well as D, are all initialized in training_hyper_params/hyper_params.ini.
- Running script (as ped1 or avenue for instances) and cd into Codes folder at first.
```
python train.py  --dataset  ped1    \
                 --train_folder  ../Data/ped1/training/frames     \
                 --test_folder  ../Data/ped1/testing/frames       \
                 --gpu  0       \
                 --iters    80000
```
- To choose the best model after every checkpoint saved run the following script and choose the one that gives the best result for the given use-case
```
python inference.py  --dataset  ped1    \
                     --test_folder  ../Data/ped1/testing/frames       \
                     --gpu  1
```

### Result
-  Model trained on 6,800 normal frames (specific to ped1 model and the dataset)
-  While the model is tested on 7,076 frames out which 3,997 frames are anomalous and the remaining 3,079 are normal frames
-  Below table shows the confusion matrix on the test set
<img src="https://github.com/ghatoleyash/Ono-Project/blob/main/Images/Confusion_Matrix.png" width=60% height=50%>

- Performance metrics: Accuracy:- 77%, Precision:- 77.56%, Recall:- 83.38%, F1:- 80.36%
- Threshold: 0.73 (based on experimental result) to distinguish the between anomalous and normal frame, if the score of the frame is below threshold value then it is tagged as anomalous frame else given as normal frame


### Reference
- Liu, Wen, et al. "Future frame prediction for anomaly detection–a new baseline." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.





