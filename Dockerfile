# Dockerfile

FROM python:3
WORKDIR /usr/src/app/
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyter

RUN mkdir /usr/src/app/videos
COPY videos/battlefield_camera.avi /usr/src/app/videos/
RUN mkdir /usr/src/app/src/
COPY background.py constants.py test.py /usr/src/app/src/
RUN mkdir /usr/src/app/notebooks/
COPY binary_matching.ipynb color_matching.ipynb matching_v7.ipynb /usr/src/app/notebooks/
WORKDIR /usr/src/app/notebooks/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]