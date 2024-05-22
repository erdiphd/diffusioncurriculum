
FROM nvidia/cuda:11.4.0-base-ubuntu20.04 

RUN apt-get update && apt-get install --no-install-recommends -y 
    

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  locales \
  git \
  libgl1-mesa-dri \
  menu \
  python \
  pip \
  mesa-utils \
  nano \
  net-tools \
  openbox \
  openssh-server \
  sudo \
  supervisor \
  terminator \
  tint2 \
  vim \
  x11-xserver-utils \
  x11vnc \
  xinit \
  xserver-xorg-video-dummy \
  xserver-xorg-input-void \
  websockify && \
  pip install supervisor-stdout && \
  apt-get -y clean

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    mpich \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN locale-gen en_US.UTF-8 en_GB.UTF-8 de_DE.UTF-8 de_DE@euro
RUN echo 'LANG=en_US.UTF-8' >> /etc/locale.conf
# RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user

# RUN adduser --disabled-password \
# --gecos '' user
# RUN adduser user sudo

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
/etc/sudoers


# Create user
RUN useradd -s /bin/bash -d /home/user/ -m -G sudo user


RUN sudo mkdir -p /home/user/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -zxf mujoco.tar.gz --directory /home/user/.mujoco \
    && rm mujoco.tar.gz


# Mini Anaconda Installation

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh \ 
    && bash Anaconda3-2023.03-1-Linux-x86_64.sh  -b -u -p /home/user/conda \
    && rm -f Anaconda3-2023.03-1-Linux-x86_64.sh 

USER user

ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so
ENV MUJOCO_GL "glfw"


#COPY outpace_official /home/user/outpace_official
WORKDIR "/home/user/"

#Incase this repo is deleted, I also backed up this repo in my gitlab repo here https://gitlab.com/erditmp/outpace_official.git
RUN sudo git clone https://github.com/jayLEE0301/outpace_official.git

RUN sudo chown -R user:user /home/user/outpace_official

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  cd /home/user/outpace_official && \
                  conda env create -f outpace.yml"



RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  cd /home/user/outpace_official && \
                  conda activate outpace && \
                  conda develop meta-nml" 

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  cd /home/user/outpace_official/meta-nml && \
                  conda activate outpace && pip install -r requirements.txt"

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  cd /home/user/outpace_official/meta-nml && \
                  conda activate outpace && pip uninstall typing -y && \
                  cd .. && chmod +x install.sh && \
                  ./install.sh" 

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  cd /home/user/outpace_official/meta-nml && \
                  conda activate outpace && pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld"

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  cd /home/user/outpace_official/meta-nml && \
                  conda activate outpace && pip install plotly==4.5.0 torch==1.13.1"



VOLUME [ "/home/user/outpace_diffusion/" ]     


ENV env_name "PointUMaze-v0"
ENV num_train_steps 3000000
ENV seed_number 0

COPY startup.sh /home/user/
ENV PATH /home/user/.conda/envs/outpace/bin:$PATH
CMD ["/usr/bin/bash", "/home/user/startup.sh"]
