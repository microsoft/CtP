FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
         libgl1-mesa-glx libglib2.0-0 && \
         pip install mmcv-full==1.2.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html && \
         pip install tensorboard && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN chmod -R a+w .