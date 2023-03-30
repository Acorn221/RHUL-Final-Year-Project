#!/bin/bash

# This script downloads the OASIS-1 dataset

for i in {1..12}
do
    filename="oasis_cross-sectional_disc${i}.tar.gz"
    url="https://download.nrg.wustl.edu/data/${filename}"
    wget "$url" -O "$filename"
done
