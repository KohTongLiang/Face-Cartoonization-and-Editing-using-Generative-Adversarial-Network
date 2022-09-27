FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

WORKDIR /workspace

COPY requirements.txt /workspace

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "/bin/bash" ]