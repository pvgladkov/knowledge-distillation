FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim \
        nano \
        tree \
        tmux \
        git \
        curl \
        autoconf \
        automake \
        cmake \
        wget \
        zip \
        unzip \
        libboost-python-dev \
        libopenblas-dev \
        libopenblas-base \
        libomp-dev \
        libjpeg-dev \
        expect \
        libtool \
        pkg-config \
        autotools-dev \
        unixodbc-dev \
        python3-pip python3-dev python3-setuptools \
        python-tk \
        locales \
        python-opencv \
    && echo 'en_US.UTF-8 UTF-8' >  /etc/locale.gen \
    && echo 'ru_RU.UTF-8 UTF-8' >> /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


ENV ASYNC_TEST_TIMEOUT=10 \
    # https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1


RUN echo $PROJECT_ROOT
COPY ./requirements.txt $PROJECT_ROOT/
RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt

RUN python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV PYTHONPATH "${PYTHONPATH}:${PROJECT_ROOT}"

CMD ["/bin/bash"]