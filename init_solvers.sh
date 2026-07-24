#!/usr/bin/env sh
# tba 1 --- # rcv1_binary
salloc --mem=64gb --cpus-per-task=16 --time=08:00:00
module load matlab
matlab -nodisplay
data="/scratch/marque6/libsvm_data/rcv1_binary.mat";
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true))
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true))
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true))
# tab 2 --- # rcv1_train

data="/scratch/marque6/libsvm_data/rcv1_train.mat";
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true, 'rbfGamma', 2.5))

data="/scratch/marque6/libsvm_data/usps.mat";
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true, 'rbfGamma', 2.5))

# tab 3 --- # news20
data="/scratch/marque6/libsvm_data/news20.mat";
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true, 'rbfGamma', 2.5))

data="/scratch/marque6/libsvm_data/ledgar.mat";
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true, 'rbfGamma', 2.5))

data="/scratch/marque6/libsvm_data/mnist.mat";
run_svm_comparison(data, struct('problem', 'mcsvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'nusvm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'svr','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l1svm','overwrite', true, 'rbfGamma', 2.5))
run_svm_comparison(data, struct('problem', 'l2svm','overwrite', true, 'rbfGamma', 2.5))
---

salloc --mem=64gb --cpus-per-task=16 --time=08:00:00
module load matlab
matlab -nodisplay
run_svm_comparison(data, struct('overwrite', true, 'problem', 'mcsvm', 'sweepGamma', [2 3 4 5], 'C', 1));
exit
exit
run_svm_comparison(data, struct('overwrite', true, 'cvGamma', [0.5 1 2 2.5 3 3.5 4 8], 'C', 1));
