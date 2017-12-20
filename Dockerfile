FROM ubuntu:xenial

RUN apt-get update
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

ADD ./images /generate_art/images/
ADD ./generate_art.py /generate_art/
ADD ./requirements.txt /generate_art/
ADD ./output /generate_art/output

RUN pip --no-cache-dir install https://github.com/mind/wheels/releases/download/tf1.4-cpu/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
RUN pip install --requirement=/generate_art/requirements.txt

CMD bash
