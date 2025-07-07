## Setup
### Dockerized setup (under construction)
```
git clone https://github.com/Kovox91/affinity_project.git

cd affinity_project/affinity_predictor

docker build -f docker/Dockerfile -t affinity-dataprep .

docker run --rm affinity-dataprep

``` 

## TODO
Next steps, for easier iterations:
- Undestand the model
- Improve the fintuning process
