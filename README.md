![logo](https://github.com/makashy/facerecognition/blob/master/images/facelogo09-01.png)

# FaceRecognition
[![working with Zharfa](https://github.com/makashy/facerecognition/blob/master/images/Screenshot.jpg)](https://www.youtube.com/watch?v=IOww6sEHmAA)
## INSTALL

1. Install [Visual Studio](https://visualstudio.microsoft.com/) (make sure to include **Desktop development with C++** module)
2. Install [CUDA](https://developer.nvidia.com/cuda-downloads)
3. Download [cuDNN](https://developer.nvidia.com/cudnn) and copy it's files to CUDA installation folder
4. Install [Anaconda](https://www.anaconda.com/distribution/) and create a conda env
5. Install [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) 

Install the followings in the env:

1. conda install -c conda-forge opencv
2. conda install -c conda-forge cmake
3. pip install dlib --verbose (or you can alternatively install it manually ⊞↓)
4. conda install -c conda-forge kivy
5. conda install -c anaconda pandas
6. conda install -c conda-forge pytables
7. conda install -c anaconda tensorflow-gpu
8. conda install -c anaconda cupy
9. pip install mpi4py

Optional packages:

* conda install -c conda-forge jupyterlab
* conda install -c anaconda pylint
* conda install -c conda-forge yapf
* conda install -c conda-forge openpyxl

⊞\
download dlib from dlib.net and unzip it in your env folder\
dlib ->(from terminal go to unziped folder of dlib and run this: python setup.py install)
