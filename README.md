# A Robust and Generalized Framework for Adversarial Graph Embedding

## Requirements
- python 3
- Tensorflow >= 1.13
- numpy
- scikit-learn
- nmslib
- annoy


## Usage
* First clone this repo.
* `pip install -r requirements.txt`
* Choose a model to run:
  * **NEGAN** (On cora): 
  ```
  cd NEGAN
  python NEGAN-lp.py --dataset cora   # for LP task
  python NEGAN-nc.py --dataset cora   # for NC task
  ```
  * **HGGAN** (On yelp):
  ```
  cd HGGAN
  # specify pre-trained embedding directory
  python hingan.py --dataset yelp --method e  # Using information from TransE (HINGAN-TE)
  ```
  * **DGGAN** (On cora):
  ```
  cd DGGAN
  python dggan.py
  ```
