FROM python:3.10
RUN apt-get update 
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . .
RUN mkdir -p /app/data/raw /app/data/processed /app/metrics /app/models && chmod -R 777 /app/data /app/models /app/metrics
# Default command (Overridden in docker-compose)
CMD ["tail", "-f", "/dev/null"]