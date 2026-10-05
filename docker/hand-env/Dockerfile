ARG BASE_IMAGE=public.ecr.aws/ubuntu/ubuntu:22.04@sha256:1eb8ef21b3345c583462171eecbda904f2497512d719e2089dc087224a91c6f0
FROM ${BASE_IMAGE} AS system
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ffmpeg git libegl1 libgl1 libglvnd0 \
    libglib2.0-0 libgomp1 libsm6 libxrender1 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

FROM system AS mint-env
RUN python3 -m venv /home/user/envs/mint
COPY requirements-mint.txt /tmp/requirements.txt
COPY check-mint-deps.py /tmp/check-mint-deps.py
# decord 0.6.0 has stale internal cp36 wheel tags; validate decoding at runtime.
RUN --mount=type=cache,id=hand-env-mint-pip,target=/root/.cache/pip \
    /home/user/envs/mint/bin/pip install pip==25.3 setuptools==69.5.1 wheel==0.45.1 \
    && /home/user/envs/mint/bin/pip install numpy==1.26.4 scipy==1.15.3 \
    && /home/user/envs/mint/bin/pip install --no-build-isolation \
       --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements.txt \
    && /home/user/envs/mint/bin/python /tmp/check-mint-deps.py \
    && /home/user/envs/mint/bin/pip freeze > /home/user/envs/mint/installed.txt

FROM system AS ego-env
RUN python3 -m venv /home/user/envs/egohandkit
COPY requirements-egohandkit.txt /tmp/requirements.txt
RUN --mount=type=cache,id=hand-env-ego-pip,target=/root/.cache/pip \
    /home/user/envs/egohandkit/bin/pip install pip==25.3 setuptools==69.5.1 wheel==0.45.1 \
    && /home/user/envs/egohandkit/bin/pip install numpy==1.26.1 scipy==1.14.1 \
    && /home/user/envs/egohandkit/bin/pip install --no-build-isolation \
       --extra-index-url https://download.pytorch.org/whl/cu126 -r /tmp/requirements.txt \
    && /home/user/envs/egohandkit/bin/pip check \
    && /home/user/envs/egohandkit/bin/pip freeze > /home/user/envs/egohandkit/installed.txt

FROM system AS runtime
ENV HOME=/home/user \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/huggingface TORCH_HOME=/tmp/torch \
    MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/ultralytics \
    PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
COPY --from=mint-env /home/user/envs/mint /home/user/envs/mint
COPY --from=ego-env /home/user/envs/egohandkit /home/user/envs/egohandkit
COPY .build/models/ /home/user/model/
COPY container-entry.sh /usr/local/bin/hand-env
COPY smoke.py /usr/local/lib/hand-env/smoke.py
COPY egl-fix/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN useradd --uid 1000 --create-home user \
    && chmod 755 /usr/local/bin/hand-env \
    && mkdir -p /home/user/data /home/user/wuji-ego-mint /home/user/EgoHandKit /home/user/roboego-hand-vis \
    && chown user:user /home/user
USER user
WORKDIR /home/user
ENTRYPOINT ["/usr/local/bin/hand-env"]
CMD ["doctor"]
