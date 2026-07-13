#!/usr/bin/env sh
salloc --mem=64gb --cpus-per-task=16 --time=04:00:00
module load matlab
matlab -nodisplay
data="/scratch/marque6/libsvm_data/rcv1_binary.mat";
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true))
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true))
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true))


data="/scratch/marque6/libsvm_data/rcv1_train.mat";
