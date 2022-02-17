# Ono-Project

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

## Dataset download for testing
cd into Data folder of project and run the shell scripts [ped1.sh, ped2.sh, avenue.sh, shanghaitech.sh](http://101.32.75.151:8181/dataset/) under the Data folder. Please manually download all datasets from ped1.tar.gz, ped2.tar.gz, avenue.tar.gz and shanghaitech.tar.gz and tar each tar.gz file, and move them in to Data folder.

## Downloading the pre-trained weights
- Download the trained models (There are the pretrained FlowNet and the trained models of the papers, such as ped1, ped2 and avenue). Please manually download pretrained models from [pretrains.tar.gz, avenue, ped1, ped2, flownet](http://101.32.75.151:8181/dataset/) and tar -xvf pretrains.tar.gz, and move pretrains into Codes/checkpoints folder.

- Running the sript (as ped2 and avenue datasets for examples) and cd into Codes folder at first. Below example is for ped1 dataset
```
python inference.py  --dataset  ped1    \
                    --test_folder  ../Data/ped1/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    checkpoints/pretrains/ped1
```

## Training from scratch 
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






