FROM mambaorg/micromamba:0.24.0

USER $MAMBA_USER:root

RUN mkdir -m 775 /home/$MAMBA_USER/.cache

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

RUN /opt/conda/bin/python3 -m spacy download en_core_web_sm

COPY --chown=$MAMBA_USER:$MAMBA_USER fastapi_app.py /fastapi_app.py

EXPOSE 5000

CMD ["python3", "/fastapi_app.py"]
