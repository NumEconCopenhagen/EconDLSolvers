#!/bin/bash

curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

bash Anaconda3-2023.09-0-Linux-x86_64.sh -b

echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc

eval "$(~/anaconda3/bin/conda shell.bash hook)"

pip install EconModel Consav
pip install torch torchvision torchaudio
pip install nvidia-ml-py
pip install line_profiler
pip install papermill

exec bash