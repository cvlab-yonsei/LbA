FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Update and install
RUN apt-get update
RUN apt-get install -y \
      git \
      vim \
      zsh \
      byobu \
      htop \
      curl \
      wget \
      locales

# Install dev
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y libgtk2.0-dev
RUN pip install -U scikit-learn
RUN pip install scikit-image

RUN apt-get install -y libpng-dev
RUN apt-get install -y libfreetype6-dev
RUN apt-get install -y libjpeg8-dev
RUN pip install matplotlib

RUN pip install --upgrade pip setuptools wheel
RUN pip install jupyter jupyterlab numpy scipy ipython pandas easydict tensorflow-gpu torchsummary tensorboard seaborn tensorboardX torchvision==0.8.2 tabulate yacs
RUN pip install opencv-python ipykernel

CMD ["/bin/bash"]