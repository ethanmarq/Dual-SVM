#!/usr/bin/env sh
salloc --mem=64gb --cpus-per-task=16 --time=04:00:00
module load matlab
matlab -nodisplay
data="/scratch/marque6/libsvm_data/rcv1_binary.mat";
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true))
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true))
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true))


data="/scratch/marque6/libsvm_data/rcv1_train.mat";

run_svm_comparison(data, struct('overwrite', true, 'sweepGamma', [2 2.5 3 3.5 4], 'C', 1));
run_svm_comparison(data, struct('overwrite', true, 'cvGamma', [0.5 1 2 2.5 3 3.5 4 8], 'C', 1));
