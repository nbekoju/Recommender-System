FROM continuumio/miniconda3

WORKDIR /app

# Update Conda
RUN conda update -n base -c defaults conda

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev

# create the conda environment
COPY environment.yml .

# Update Conda and create the Conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "recommender_fast_api", "/bin/bash", "-c"]

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port that app runs on
EXPOSE 8000

# Activate the environment and run the FastAPI application
CMD ["conda", "run", "--no-capture-output", "-n", "recommender_fast_api", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
