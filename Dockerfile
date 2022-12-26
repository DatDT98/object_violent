FROM jrottenberg/ffmpeg:4.3-ubuntu1804

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get -y install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt install libgl1-mesa-glx -y

RUN apt install git -y
EXPOSE 50051
# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN /usr/bin/python3.6 -m pip install --upgrade pip


COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /forgot-object

WORKDIR /forgot-object
