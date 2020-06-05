# Dockerfile

FROM python:3
WORKDIR /usr/src/app/
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyter

WORKDIR /usr/src/app
RUN mkdir /usr/src/app/videos
COPY videos/battlefield_camera.avi /usr/src/app/videos/
COPY background.py constants.py custom_detector.py main.py test.py binary_matching.ipynb color_matching.ipynb matching_v7.ipynb ./

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]