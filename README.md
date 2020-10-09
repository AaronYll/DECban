## DECban: Full-length circRNA-RBP interaction sites prediction by using Double Embedding Cross branch attention Network


If you want to reproduce this work, please follow these steps. 
1. Prepare a conda virtual environment:
```
conda create -n decban python=3.6.2
conda activate decban
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

2. Download dataset from [Cloud disk](https://pan.baidu.com/s/1W2dau1uSto1jkDFvH0ZX_g) (password:o6tl). Unzip this file to `/DECban/prep` directory.

3. Run demo:
```
cd script
python Train.py
```

Then model will start training, training metrics will be saved in directory `/DECban/metric/board`, model checkpoint will be saved in dicrectory `/DECban/metric/checkpoint`.

