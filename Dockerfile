# syntax=docker/dockerfile:1.4
FROM python:3.10 AS builder

# Add build dependencies
# RUN apk add --no-cache g++ gfortran build-base openblas openblas-dev
RUN apt update
RUN apt install -y git g++ gfortran build-essential

WORKDIR /app

COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt
COPY . /app

ENTRYPOINT ["python3"]
CMD ["py/app.py"]

FROM builder as dev-envs

RUN addgroup -S docker && \
    adduser -S --shell /bin/bash --ingroup docker vscode

# Assuming the necessary Docker tools are present in the directory /docker-tools,
# you can copy them into the image as follows. Adjust the source path as needed.
# COPY --from=/docker-tools / /usr/local/bin/

# If Docker tools are to be copied from another image, use the following line instead:
COPY --from=gloursdocker/docker / /usr/local/bin/
