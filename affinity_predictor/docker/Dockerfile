FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

# Use dedicated writable working dir for pip builds
WORKDIR /opt/app
COPY --chown=mambauser:mambauser . /opt/app

# Update micromamba to avoid bugs
RUN micromamba update -n base -c conda-forge micromamba

# Install PyMOL or other dependencies
RUN micromamba clean --all --yes \
 && micromamba install -y -n base -c conda-forge pymol-open-source \
 && micromamba clean --all --yes

# Switch to app user
USER mambauser

# Ensure everything is writable (sometimes submodules stay root-owned)
RUN chmod -R u+rwX /opt/app

# Install your package
RUN pip install --upgrade pip setuptools \
 && pip wheel /opt/app -w /tmp/wheels \
 && pip install /tmp/wheels/affinity_predictor-*.whl

WORKDIR /opt/app
ENTRYPOINT ["kedro", "run"]
