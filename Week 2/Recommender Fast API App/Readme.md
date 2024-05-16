## Required files

All the following files are present in **chapter 7 of Hands on Recommender System folder in this repository**
- cosine_similarities_sbert.pkl : cosine similarties for content based recommender
- indices.pkl
- movie_title.pkl
- svd_model.pkl : for collaborative filtering


## To run the app locally

Build the docker image
```
docker build -t fastapi-conda-app .
```

Run the docker container
```
docker run -d -p 8000:8000 fastapi-conda-app
```
