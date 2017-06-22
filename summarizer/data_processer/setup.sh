#!/bin/sh

echo "Step: Cleaning and making corpus structure"
path=$PWD
data_path="$path/../data"

mkdir -p "$data_path/processed"

python make_DUC.py --data_set=DUC2004 --data_path=$data_path --parser_type=parse

