FROM mambaorg/micromamba:0.24.0

USER $MAMBA_USER:root

RUN mkdir -m 775 /home/$MAMBA_USER/.cache

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER src/*.py /src/

EXPOSE 8501

CMD ["streamlit", "run", "/src/streamlit_app.py"]
