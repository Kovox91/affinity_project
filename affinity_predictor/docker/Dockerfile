FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

# Clone project and its submodules
RUN git clone --branch remote_prep git clone https://github.com/Kovox91/affinity_project.git /opt/app \
 && cd /opt/app \
 && git submodule update --init --recursive

WORKDIR /opt/app

# Update micromamba
RUN micromamba update -n base -c conda-forge micromamba

# Install PyMOL
RUN micromamba clean --all --yes \
 && micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# Set permissions before switching user
RUN chown -R mambauser:mambauser /opt/app

USER mambauser

RUN pip install --upgrade pip setuptools \
 && pip wheel /opt/app -w /tmp/wheels \
 && pip install /tmp/wheels/affinity_predictor-*.whl

ENTRYPOINT ["kedro", "run"]
