FROM python:3.6.13

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt

RUN chmod +x start.sh

CMD [ "./start.sh"]