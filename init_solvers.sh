#!/usr/bin/env sh
salloc --mem=128gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
run_svm_comparison("/scratch/marque6/libsvm_data/rcv1_binary.mat", struct('problem', 'nusvm','overwrite', true))
run_svm_comparison("/scratch/marque6/libsvm_data/rcv1_binary.mat", struct('problem', 'l1svm','overwrite', true))
run_svm_comparison("/scratch/marque6/libsvm_data/rcv1_binary.mat", struct('problem', 'l2svm','overwrite', true))
run_svm_comparison("/scratch/marque6/libsvm_data/rcv1_binary.mat", struct('problem', 'svr','overwrite', true))
run_svm_comparison("/scratch/marque6/libsvm_data/rcv1_binary.mat", struct('problem', 'mcsvm','overwrite', true))


run_svm_comparison("/scratch/marque6/libsvm_data/a9a_binary.mat", struct('overwrite', true))

run_svm_comparison("/scratch/marque6/libsvm_data/rcv1_train.mat", struct('problem', 'mcsvm', 'overwrite', true))
