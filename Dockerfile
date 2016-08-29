FROM python:3.5.1
RUN apt-get update && apt-get install -y \
        && mkdir /opt/resonances-ml
ADD . /opt/resonances-ml
RUN pip install -e /opt/resonances-ml
WORKDIR /opt/resonances-ml
ENTRYPOINT ["python", "-m", "resonancesml"]
