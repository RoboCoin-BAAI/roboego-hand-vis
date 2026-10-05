ARG HAND_BASE=roboego-hand-env:dev
FROM ${HAND_BASE}

USER root
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
USER user
