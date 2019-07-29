FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Craig Citro <craigcitro@google.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        python3-tk \
        git-core \
        apt-transport-https 
        
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Set up our notebook config.
COPY . ./clusterfiles
COPY deployed_checkpoints/text-class.10-0.42.pkl ./clusterfiles/deployed_checkpoints/
COPY deployed_checkpoints/text-class.10-0.42.hdf5 ./clusterfiles/deployed_checkpoints/

RUN pip3 install numpy \
        && \
    pip3 --no-cache-dir install -r ./clusterfiles/requirements.txt 

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt

# matplotlib config (used by benchmark)
#RUN mkdir -p /root/.config/matplotlib
#RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow GPU version.
# RUN pip3 --no-cache-dir install \
#     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# RUN ln -s /usr/bin/python3 /usr/bin/python#

RUN ls /opt #find / -name "libmsodbcsql*"

# For CUDA profiling, TensorFlow requires CUPTI.
#ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR ./clusterfiles

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN echo $LC_ALL &&\
    echo $LANG

#RUN chmod +x start_flask.sh && \
#    sync

RUN chmod +x start_gRPC.sh && \
    sync

# this is the gRPC server command
CMD ./start_gRPC.sh

# this is the REST server command
#CMD ./start_flask.sh