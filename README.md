# Project Setup and Guide

## Initial Setup

**Create a Virtual Environment, activate it, and install requirements**
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
python src/main.py
```
Initial run reveals Random Forest as the best model with an F1 score of 0.9991.

## Steps and Features
### 1. Exploratory Data Analysis (EDA)
   - Added an EDA code to discover more about the data.
   - Findings can be reviewed in the EDA file.

### 2. Experimenting with Models

- I tested boosting algorithms but they did not outperform Random Forest.
Random Forest which is an ensemble method, provided the best results.
This was better than any other ensemble methods such as HistogramBoost

### 3. MLFlow Integration

- MLFlow has been set up in the codebase to track and compare model performance.
To view model metrics in MLFlow UI when using Docker, follow these steps:

`docker container ls`

- Find the container named relaxy-ai-engineer-exam-question_relaxy-exam and use the container ID.
```
docker exec -it [CONTAINER ID] sh
mlflow ui --host 0.0.0.0
```
To run locally (without Docker), simply execute the following command in shell:

mlflow ui

### 4. Setting Up EC2
- Followed the standard procedure to create an EC2 instance on AWS.
- Current setup supports local pipeline execution. To run on EC2:
    Modify the docker-compose file:
```
services:
  relaxy-exam:
    build:
      context: .
      dockerfile: Dockerfile
```
To:
```
services:
  relaxy-exam:
    build:
      image: docker.io/ayhay/relaxy-exam
```

### 6. Using the Docker Image and running the pipeline

- Build and push your own image if you needed to make any changes to the code:
```
docker build
docker push
```
- I have already pushed the docker image into docker-hub so you can pull it using
```docker pull docker.io/ayhay/relaxy-exam```
- You can either connect to the EC2 instance shell or your local shell if you are testing locally and run:
`docker compose up -d`
- The docker-compose file has already set up prometheus for metrics tracking and alert notification if something goes wrong, 
and it also supports using the existing image, so nothing else needs to be done. 

### 7. Testing the Endpoints

- After setting up the project using docker-compose, you can test the endpoints:
```
Local Testing
    URL: http://127.0.0.1:8080

Testing on EC2
    Replace the 127.0.0.1 part with the AWS-provided public IP or domain.
```
- Implemented Endpoints
```
    /metrics (GET)
    Returns Prometheus metrics logs for analyzing:
        - Infrastructure details
        - CPU usage
        - Event counters, etc.

    /health (GET)
    - Checks if the best-performing model is active and available.

    /predict (POST)
    - Predicts whether a user will get a loan based on input data.
