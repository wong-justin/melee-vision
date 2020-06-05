# Dockerfile

FROM python:3
WORKDIR /usr/src/app/
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyter

RUN mkdir /usr/src/app/videos
COPY videos/battlefield_camera.avi /usr/src/app/videos/
RUN mkdir /usr/src/app/src/
COPY src/background.py src/constants.py src/test.py /usr/src/app/src/
RUN mkdir /usr/src/app/notebooks/
COPY notebooks/binary_matching.ipynb notebooks/color_matching.ipynb notebooks/matching_v7.ipynb /usr/src/app/notebooks/
WORKDIR /usr/src/app/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]