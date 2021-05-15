# Home Assignment #2: Building REST service with FastAPI

## Run a service
```cmd
# go to project directory:
cd ./online_inference

# then build and run docker container:
docker-compose up -d --build

# make sure that the service is working:
docker-compose logs

  Attaching to online_inference_api
  api  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  api  | INFO:     Started reloader process [1] using statreload
  api  | INFO:     Started server process [10]
  api  | INFO:     Waiting for application startup.
  api  | INFO:     Application startup complete.

```
## API endpoints
- `/` - GET, root endpoint
- `/health` - GET, check whether service is ready for making predictions
- `/predict` - POST, predict the probability of heart disease
- `/docs`- automatically generated interactive API documentation
  
## Request the service
There is a command-line interface (path: `online_inference/api/request_service.py`) which makes it possible to send some POST-requests to the service and print predicted probability to `sys.stdout`.<br>
There are the following options for this CLI:
- `--count` - OPTIONAL, define the number of requests to the service. By default, `count=1`
- `--random-data` - OPTIONAL, describe whether to use random data (generated synthetically). Whithout this option uses random string from real dataset [Heart Disease UCI data](https://www.kaggle.com/ronitf/heart-disease-uci)

For example, to send five POST requests with random data use the following command:
```cmd
python online_inference/api/request_service.py --count 5 --random-data
```

## Tests
There are some test cases for this API. To run the tests use the following command:
```cmd
pytest online_inference/api/app/tests
```

## Docker image on Docker Hub
There is a docker image with the service available on Docker Hub: `garistvlad/heart-classifier-api`.<br>
The compressed size of the optimized image is `229.42 MB`.
