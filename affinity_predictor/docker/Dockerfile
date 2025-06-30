# docker/Dockerfile

FROM mambaorg/micromamba:1.5.6

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=base \
    PYTHONNOUSERSITE=1

# Create working directory inside the container
WORKDIR /app

# Copy everything into the image (minimal .dockerignore is ideal)
COPY . /app

# set up proper permissions
RUN chown -R $MAMBA_USER /app


# Install PyMOL via micromamba (conda-forge channel)
RUN micromamba install -y -n base -c conda-forge \
    pymol-open-source \
    && micromamba clean --all --yes

# Install Python/Kedro deps via pip
RUN pip install --upgrade pip \
 && pip install .

# Default: run Kedro pipeline
ENTRYPOINT ["kedro", "run"]
