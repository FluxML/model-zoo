FROM gitpod/workspace-full

USER gitpod

# Install Julia
RUN sudo apt-get update \
    && sudo apt-get install -y \
        libatomic1 \
        gfortran \
        perl \
        wget \
        m4 \
        pkg-config \
        julia \
    && sudo rm -rf /var/lib/apt/lists/*
