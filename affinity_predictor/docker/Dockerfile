FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

# Install git
RUN micromamba install -y -n base -c conda-forge git \
 && micromamba clean --all --yes

# Clone into a writable location
RUN git clone --branch remote_prep https://github.com/Kovox91/affinity_project.git /home/mambauser/app \
 && cd /home/mambauser/app \
 && git submodule update --init --recursive \
 && rm -rf /home/mambauser/app/.git

# Update micromamba
RUN micromamba update -n base -c conda-forge micromamba

# Install dependencies
RUN micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# Switch to app user
USER mambauser
WORKDIR /home/mambauser/app/affinity_predictor

RUN pip install --upgrade pip setuptools \
 && pip wheel . -w /tmp/wheels \
 && pip install /tmp/wheels/affinity_predictor-*.whl


ENTRYPOINT ["kedro", "run"]
