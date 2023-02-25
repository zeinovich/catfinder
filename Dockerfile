FROM python:3.10

WORKDIR /home/catfinder

COPY ./requirements.txt /home/catfinder/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY . /home/catfinder

CMD ["python", "bot/bot.py"]