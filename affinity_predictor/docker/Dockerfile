FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

# Clean separate working dir for pip install
WORKDIR /build
COPY . /opt/app

# Install dependencies
RUN micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# Install your project — build from /opt/app but write nowhere restricted
RUN pip install --upgrade pip setuptools \
 && pip install -e /opt/app

# Set actual workdir for container
WORKDIR /opt/app
ENTRYPOINT ["kedro", "run"]
