FROM ubuntu:xenial

RUN apt-get update
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

ADD ./requirements.txt /generate_art/

RUN pip --no-cache-dir install https://github.com/mind/wheels/releases/download/tf1.4-cpu/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
RUN pip install --requirement=/generate_art/requirements.txt

CMD bash
