FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
RUN apt-get update && apt-get install -y \
  g++ \
  gcc \
  make \
  && rm -rf /var/lib/apt/lists/*

RUN gcc --version
# ENV CUDA_HOME=/usr/local/cuda-10.1
ENV CUDA_HOME=/usr/local/cuda
COPY . /work
RUN cd /work
WORKDIR /work
RUN pip install matplotlib --ignore-installed certifi
RUN pip install tensorflow==1.14
RUN pip install tensorboard
RUN pip install opencv-python 
RUN pip install plyfile
RUN pip install trimesh==2.35.39
RUN pip install networkx==2.2
RUN cd pointnet2 && python setup.py install
RUN CUDA_VISIBLE_DEVICES=0 python models/votenet.py
