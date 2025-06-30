FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

WORKDIR /opt/app
COPY . /opt/app

# Install PyMOL and Python deps
RUN micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# Fix: install setuptools before pip install
RUN pip install --upgrade pip setuptools \
 && pip install --no-build-isolation .

ENTRYPOINT ["kedro", "run"]
