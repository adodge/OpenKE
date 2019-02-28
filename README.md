# OpenKE
An Open-source Framework for Knowledge Embedding.

## Overview

This is a fork of the [OpenKE](https://github.com/thunlp/OpenKE) project, with
the intention of exposing a more standard and reusable python module.  I'll do
my best to sync up improvements to the master version, as well as submit pull
requrests for improvements made to this fork.

## Installation

```
pip3 install git+https://github.com/adodge/OpenKE.git
```

## Data

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
generation should be made part of the DataLoader object at some point.

## Quick Start

### Training

An example script is provided that uses the data loader, trains a model, and
dumps it to a file.

```
python3 openke_train.py
```

### Testing

This script loads the model from the other script and applies the two methods
of testing that are provided by the framework: link prediction and triple
classification.

```
python3 openke_test.py
```

#### Link Prediction

TODO Read through the test code to find out what these tests are actually
testing.

#### Triple Classificiation

TODO Read through the test code to find out what these tests are actually
testing.

## Citations

### OpenKE

This is a fork of the OpenKE project.  Here is the information on the original
project:

More information is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)

If you use the code, please cite the following
[paper](http://aclweb.org/anthology/D18-2024):

```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```
