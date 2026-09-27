ARG BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:22.04@sha256:1eb8ef21b3345c583462171eecbda904f2497512d719e2089dc087224a91c6f0
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build ccache git ca-certificates \
    libopencv-dev libeigen3-dev libceres-dev \
    libboost-system-dev libboost-filesystem-dev libboost-thread-dev libboost-date-time-dev \
    python3 python3-numpy python3-scipy python3-opencv python3-yaml python3-pytest ffmpeg \
    && dpkg-query -W > /usr/local/share/vio-packages.txt \
    && rm -rf /var/lib/apt/lists/*
ENV HOME=/tmp/vio-home PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 PYTHONPATH=/home/user/roboego-hand-vis \
    CCACHE_DIR=/home/user/build/ccache
COPY entry.sh /usr/local/bin/vio-env
RUN chmod 755 /usr/local/bin/vio-env
WORKDIR /home/user/roboego-hand-vis
ENTRYPOINT ["/usr/local/bin/vio-env"]
CMD ["doctor"]
