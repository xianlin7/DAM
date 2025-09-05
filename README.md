# DAM
This repo is the official implementation for DAM

## Installation
`python=3.8.20`, `pytorch=1.10.0`, and `torchvision=0.11.0` are used in DAM.

1. Clone the repository.
    ```
    git clone https://github.com/xianlin7/DAM.git
    cd DAM
    ```
2. Create a virtual environment for DAM and activate the environment.
    ```
    conda create -n DAM python=3.8
    conda activate DAM
    ```
3. Install Pytorch [`pytorch=1.10.0`] and TorchVision [`torchvision=0.11.0`].
   (you can follow the instructions [here](https://pytorch.org/get-started/locally/))
5. Install other dependencies.
  ```
    pip install -r requirements.txt
  ```


## Data
- CT5M-Gref consists of 5 publicly-available datasets:
    - [INSTANCE](https://instance.grand-challenge.org)
    - [Covid-19-20](https://covid-segmentation.grand-challenge.org/COVID-19-20/)
    - [DecathlonZ Task07](http://medicaldecathlon.com/)
    - [WORD](https://github.com/hilab-git/word)
    - [Adrenal-ACC-Ki67](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93257945)
- All images were sampled from [CT5M](https://ieeexplore.ieee.org/abstract/document/10746534), where the data is in 2D PNG format.
- We have provided some data examples. Please refer to the file folder [./data_examples](https://github.com/xianlin7/DAM/tree/main/data_examples).\
- The relevant information of your data should be configured in [./utils/config.py](https://github.com/xianlin7/DAM/blob/main/utils/config.py).
- We will release the dataset in the future 🌝.
## Training
Once you have the data ready, you can start training the model.
```
cd "/home/...  .../DAM/"
python train.py --modelname DAM_GMIRS --task <your dataset config name>
```
## Testing
Do not forget to set the load_path in [test.py](https://github.com/xianlin7/DAM/blob/main/test.py) before testing.
```
python test.py --modelname DAM_GMIRS --task <your dataset config name>
```
