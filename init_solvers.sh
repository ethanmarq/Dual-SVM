#!/usr/bin/env sh
salloc --mem=128gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
run_svm_comparison("/scratch/marque6/libsvm_data/rcv1_binary.mat")
