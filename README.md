# OpenKE
An Open-source Framework for Knowledge Embedding.

## Overview

This is a fork of the [OpenKE](https://github.com/thunlp/OpenKE) project, with
the intention of exposing a more standard and reusable python module.  I'll do
my best to sync up improvements to the master version, as well as submit pull
requests for improvements made to this fork.

Major differences from the original project:

* Removed the Config object.
  * Model hyperparameters are moved into the model initializer.
  * Data loading and interfacing with the C library is moved into the
    DataLoader class.
  * Experimental setup is left up to the user of the module.  Example scripts
    are provided.
* Models act a bit more like Keras layers, where initializing the model
  allocates the parameters, and then the model exposes methods for constructing
  parts of a computation graph.  This should make it easier to reuse them as
  part of a larger network.
* The python wrapper for the C library should be a bit more opaque, to provide
  an easy interface for loading and sampling data without worrying too much
  about how it's allocating memory, etc.

## Installation

```bash
pip3 install git+https://github.com/adodge/OpenKE.git
```

## Data

Several datasets are included in the benchmarks directory.  TODO Find out where
they came from and put in a README.

A dataset consists of the following:

### Entity/Relation ID maps

Mapping from entity/relation names to IDs is done with TSV files.  The first
line is the number of entries in the file, and the rest have the columns: name
(string), id (integer)

* entity2id.txt
* relation2id.txt

These files aren't actually used by any of the existing code, except to
discover the number of entities and relations.  IDs should range from 0 to N
(exclusive), where N is the number of entities/relations.


### Triples

Triples are stored as a TSV file, where the first line is the number of triples
in the file and the rest have the columns: head, tail, relation, where these
are represented by entity and relation IDs.

There are three files with this format:

* train2id.txt
* valid2id.txt
* test2id.txt

valid2id.txt and test2id.txt are unused during training, and are only necessary
if you're using the testing functions.

### Type constraints

This is a generated file, containing the domain and range of each relation in a
fairly esoteric format.  It's the output of the `n-n.py` script, but its
generation should be made part of the DataLoader object at some point. TODO

## APIs

### DataLoader

The DataLoader class offers an optimized, consistent method of loading the
training and test data, and also generating negative examples.  This is taken
from the OpenKE project, with a different python interface.

At its most basic, the DataLoader is initialized with the path to the dataset
directory.  For example:

```python
import OpenKE
data = OpenKE.DataLoader(data_path='./benchmarks/WN18RR')
```

The main interface it exposes is the `.sample()` method.  This returns four
numpy arrays:

```python
h,t,r,y = data.sample()
```

These are in the appropriate format for sending in as training data for a
model.

The DataLoader also exposes useful information about the data, like the number
of entities and relations, the size of the different corpus partitions, etc.
This is useful for choosing the sizes of arrays in your model.  For example, to
make sure the embedding matrices are the right size:

```python
model = OpenKE.models.TransE(
    n_entities=data.n_entities,
    n_relations=data.n_relations)
```

### Models

All models have at least the following initialization options:

* n\_entities
* n\_relations

These are most conveniently gotten from the DataLoader object properties of the
same name.

Creating the model object defines the variables for the parameters, so this
needs to be done inside a tensorflow graph/session context:

```python
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default()
        model = OpenKE.models.TransE(
            n_entities=data.n_entities,
            n_relations=data.n_relations)
```

To add the model to your computation graph, call the methods on the model
object, passing in inputs (as tensorflow nodes) and returning outputs (also
tensorflow nodes).

All models implement at least the following methods:

* loss
* predict

#### Analogy

TODO What is this model?  A combination of ComplEx and DistMult?

Additional hyperparameters:
* hidden\_size
* lmbda

#### ComplEx
* [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)

Additional hyperparameters:
* hidden\_size
* lmbda

NOTE: Needs to be trained with a much higher alpha (like 0.1)

#### DistMult
* [DistMult](https://arxiv.org/pdf/1412.6575.pdf)

Additional hyperparameters:
* hidden\_size
* lmbda

NOTE: Needs to be trained with a much higher alpha (like 0.1)

#### HolE
* [HolE](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)

Additional hyperparameters:
* hidden\_size
* margin

NOTE: Needs to be trained with a much higher alpha (like 0.1)

#### RESCAL
* [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)

Additional hyperparameters:
* hidden\_size
* margin

NOTE: Needs to be trained with a much higher alpha (like 0.1)

#### TransD
* [TransD](http://anthology.aclweb.org/P/P15/P15-1067.pdf)

Additional hyperparameters:
* hidden\_size
* margin

#### TransE
* [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

Additional hyperparameters:
* hidden\_size
* margin

#### TransH
* [TransH](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546)

Additional hyperparameters:
* hidden\_size
* rel\_size
* margin

#### TransR
* [TransR](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/)

Additional hyperparameters:
* hidden\_size
* ent\_size
* rel\_size
* margin

### Using the Model

TODO implement/document these

#### Extracting Embeddings

A Model object exposes a property called `.parameters` which will be a
dictionary mapping names onto numpy arrays.  The parameters will be particular
to the specific model, but there will usually be parameters named
`ent_embeddings` and `rel_embeddings` which contain the embeddings for entities
and relations.

```python
import OpenKE
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        model = OpenKE.models.Model.load('path_to_model.pickle')

        params = model.parameters
        entity_embeddings = params['ent_embeddings']

        # You can also set the model parameters
        model.parameters = {'ent_embeddings': np.zeros_like(entity_embeddings)}
```

#### Predict Head/Tail Entity

#### Predict Relation

#### Classify Triple

## Quick Start

### Training

An example script is provided that uses the data loader, trains a model, and
dumps it to a file.

```bash
python3 openke_train.py
```

TODO Add cli options

### Testing

This script loads the model from the other script and applies the two methods
of testing that are provided by the framework: link prediction and triple
classification.

```bash
python3 openke_test.py
```

TODO Add cli options

#### Link Prediction

Link prediction aims to predict the missing h or t for a relation fact triple
(h, r, t). In this task, for each position of missing entity, the system is
asked to rank a set of candidate entities from the knowledge graph, instead of
only giving one best result. For each test triple (h, r, t), we replace the
head/tail entity by all entities in the knowledge graph, and rank these
entities in descending order of similarity scores calculated by score function
fr. we use two measures as our evaluation metric:

* *MR* : mean rank of correct entities;
* *MRR*: the average of the reciprocal ranks of correct entities;
* *Hit@N* : proportion of correct entities in top-N ranked entities.

TODO Describe the constraint stuff
TODO Return the output from the DataLoader method, instead of printing to
     stdout
TODO Reset the global variable in C after this call, so we can call it multiple
     times without restarting python.

#### Triple Classificiation

Triple classification aims to judge whether a given triple (h, r, t) is correct
or not. This is a binary classification task. For triple classification, we set
a relationspecific threshold δr. For a triple (h, r, t), if the dissimilarity
score obtained by fr is below δr, the triple will be classified as positive,
otherwise negative. δr is optimized by maximizing classification accuracies on
the validation set.


## Citations

### OpenKE

This is a fork of the [OpenKE](https://github.com/thunlp/OpenKE) project.

If you use the code, please cite the following
[paper](http://aclweb.org/anthology/D18-2024):

```latex
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```

### Embedding Models

* [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)
* [DistMult](https://arxiv.org/pdf/1412.6575.pdf)
* [HolE](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)
* [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)
* [TransD](http://anthology.aclweb.org/P/P15/P15-1067.pdf)
* [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
* [TransH](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546)
* [TransR](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/)

### Benchmark Data

TODO
