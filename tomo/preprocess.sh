#!/bin/bash

set -ex

cat corpora/*.txt | tunimi > preprocessed.txt

