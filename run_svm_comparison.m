function results = run_svm_comparison(matFile, opts)
%RUN_SVM_COMPARISON  Benchmark dual projected-gradient SVM solvers.
%
%   RESULTS = RUN_SVM_COMPARISON(MATFILE) trains one SVM variant on the data in
%   MATFILE with the proposed projected-gradient dual solver and with reference
%   solvers (LIBSVM SMO, kernelized dual coordinate descent), and writes a
%   suboptimality-versus-time figure to
%
%       <opts.outRoot>/<dataset>/<problem>_<costMode>.png
%
%   RESULTS = RUN_SVM_COMPARISON(MATFILE, OPTS) overrides the defaults; see
%   FILL_DEFAULT_OPTS for the full list. The options that matter most:
%
%       opts.problem     'l1svm' | 'l2svm' | 'nusvm' | 'mcsvm' | 'svr'
%       opts.costMode    'none' | 'cost'   (per-class C_i weighting)
%       opts.biasMode    'constrained' | 'none'
%       opts.timeLimit   solver wall-clock budget, seconds
%
%   The kernel is RBF throughout; opts.rbfGamma sets its width. The Gram is
%   always materialized (n <= opts.explicitKernelMaxN is enforced), because
%   every coordinate baseline needs cached kernel columns and a matrix-free
%   matvec was measured 55-63x slower than dense on the benchmarked data.
%
%   Every solver is timed on the same clock, which is paused while the
%   objective is recorded, and which starts pre-charged with the setup cost
%   that solver actually incurs (see SETUP ACCOUNTING below).
%
%   BIAS AND THE DUAL EQUALITY CONSTRAINT
%       The l1/l2/svr duals carry an equality constraint (<alpha,y> = 0 or
%       1'beta = 0) that comes from the primal bias term b. Single-coordinate
%       descent cannot move on such a dual, so opts.biasMode selects the
%       formulation -- and with it the TRACK:
%
%         'none'         Track A. No bias at all (LIBLINEAR -B -1, and the
%                        setting used by Hsieh et al. 2008). Box-only dual:
%                        PG and the coordinate baselines apply; LIBSVM skips
%                        itself, since it structurally enforces the equality
%                        and would solve a different problem (f*_eq >= f*_box).
%         'constrained'  Track B. Keep the equality. PG bisects for the
%                        multiplier; the coordinate baselines skip themselves;
%                        LIBSVM applies.
%
%       nusvm is exempt: its two class-mass equalities define nu and cannot be
%       reformulated away, so its coordinate baseline uses same-class PAIRS.
%       mcsvm is exempt: Crammer-Singer has no bias term, and its equality is
%       per-row, so block coordinate descent applies as-is.
%
%   SETUP ACCOUNTING
%       ker.gramTime  building K.       Needed by PG and by every coordinate
%                                       baseline that reads ker.K. Shared.
%       ker.sig1Time  eigs on ker.mul.  Needed by PG only -- no coordinate
%                                       method reads ker.sig1.
%
%       PG solvers are charged P.setupPG = gramTime + sig1Time.
%       Coordinate and SMO baselines that read ker.K are charged
%       P.setupGram = gramTime.
%       LIBSVM builds its own kernel cache inside svmtrain (-t 2), so nothing
%       is pre-charged; with -t 4 (l2svm) it needs a precomputed Gram, which
%       is timed and added to every recorded point.
%
%   COST AXIS (machine-independent)
%       Alongside wall-clock, every instrumented history records hist.cols:
%       cumulative kernel-column reads weighted by right-hand-side width.
%       One coordinate update = 1, a pair step = 2, an mcsvm row update = K,
%       a dense matvec K*a = n, K*A with an n x K RHS = nK, a sparse-delta
%       matvec = nnz(delta), and each eigs matvec = n. Dividing by the cost
%       of one full gradient (n, or nK for mcsvm) gives gradient equivalents,
%       under which one coordinate epoch == one matvec by construction.
%       LIBSVM is a black box, so its cols are NaN. The unit ignores memory
%       locality: BLAS streams a matvec while a coordinate epoch scatters n
%       separate column reads, so this axis flatters the coordinate methods
%       exactly as wall-clock flatters PG -- the two bracket the
%       implementation-independent answer.
%
%   REFERENCES
%       Hsieh, Chang, Lin, Keerthi, Sundararajan (ICML 2008).
%           A dual coordinate descent method for large-scale linear SVM.
%       Mangasarian & Musicant (IEEE TNN 1999).
%           Successive overrelaxation for support vector machines.
%       Keerthi, Sundararajan, Chang, Hsieh, Lin (2008).
%           A sequential dual method for large scale multi-class linear SVMs.
%       Ho & Lin (JMLR 2012).
%           Large-scale linear support vector regression.
%       Chang & Lin (2011).  LIBSVM: a library for support vector machines.
%
%   See also SOLVE_L1L2, SOLVE_SVR, SOLVE_MCSVM, SOLVE_NUSVM.

    addpath('./libsvm-336/matlab');

    if nargin >= 2 && isfield(opts, 'sweepGamma')
        matFile = "/scratch/marque6/libsvm_data/rcv1_binary.mat";
        results = gamma_sweep(matFile, opts);
        return;
    end
    if nargin >= 2 && isfield(opts, 'cvGamma')
        results = cv_gamma_sweep(matFile, opts);
        return;
    end

    if nargin < 2
        opts = struct();
    end
    opts = fill_default_opts(opts);
    rng(opts.seed);

    % ---- output path -----------------------------------------------------
    [~, datasetName] = fileparts(matFile);
    figDir  = fullfile(opts.outRoot, datasetName);
    figPath = fullfile(figDir, sprintf('%s_%s_%s.png', ...
                       opts.problem, opts.costMode, opts.biasMode));

    if exist(figPath, 'file') && ~opts.overwrite
        fprintf('[skip] %s exists (set opts.overwrite = true to redo)\n', figPath);
        results = struct('skipped', true, 'figPath', figPath);
        return;
    end
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end

    % ---- data and problem ------------------------------------------------
    [X, y]       = load_xy_from_mat(matFile);
    [X, y, meta] = preprocess_xy(X, y, opts);
    P            = make_problem(y, meta, opts);

    fprintf('%s | n = %d, d = %d | problem = %s | costMode = %s | biasMode = %s | C = %g\n', ...
            datasetName, size(X, 1), size(X, 2), opts.problem, opts.costMode, ...
            opts.biasMode, opts.C);

    % ---- kernel operator -------------------------------------------------
    ker = make_kernel_op(X, y, P, opts);
    report_conditioning(ker);

    P.setupGram = ker.gramTime;                   % billed to every ker.K reader
    P.setupPG   = ker.gramTime + ker.sig1Time;    % PG also pays for sigma_1
    P.setupGramCols = ker.gramCols;               % same split, in column units
    P.setupPGCols   = ker.gramCols + ker.sig1Cols;
    fprintf('setup: Gram %.2f s (shared) + sigma_1 %.2f s (PG only)\n', ...
            ker.gramTime, ker.sig1Time);

    % ---- proposed solver -------------------------------------------------
    switch P.name
        case {'l1svm', 'l2svm'}
            propOut = solve_l1l2(ker, y, P, opts);
        case 'svr'
            propOut = solve_svr(ker, y, P, opts);
        case 'mcsvm'
            propOut = solve_mcsvm(ker, y, P, opts);
        case 'nusvm'
            propOut = solve_nusvm(ker, y, P, opts);
    end
    propLabel = sprintf('PG Dual (%s)', P.name);

    % ---- reference solvers -----------------------------------------------
    % Flat table per variant. Track routing is done by the guards inside each
    % baseline: the DCD baselines skip themselves when P.hasEq (Track B), and
    % baseline_libsvm_sweep skips itself when ~P.hasEq on l1/l2/svr (Track A).
    switch P.name
        case 'mcsvm'
            cand = { baseline_dcd_mcsvm(ker, X, y, P, opts), 'CS sequential dual (Keerthi et al. 2008)'
                     baseline_smo_mcsvm(ker, X, y, P, opts), 'CS SMO (max-violating pair)' };
        case 'nusvm'
            cand = { baseline_pcd_nusvm(ker, X, y, P, opts),    'Pairwise dual CD (same-class SMO)'
                     baseline_libsvm_sweep(ker, X, y, P, opts), 'LIBSVM SMO' };
        case 'svr'
            cand = { baseline_dcd_svr(ker, X, y, P, opts),      'Kernel dual CD (SVR)'
                     baseline_libsvm_sweep(ker, X, y, P, opts), 'LIBSVM SMO' };
        case {'l1svm', 'l2svm'}
            cand = { baseline_dcd_binary(ker, X, y, P, opts),   'Kernel dual CD / SOR'
                     baseline_libsvm_sweep(ker, X, y, P, opts), 'LIBSVM SMO' };
        otherwise
            cand = cell(0, 2);
    end

    baseOuts   = {};
    baseLabels = {};
    for c = 1:size(cand, 1)
        if ~cand{c, 1}.skipped
            baseOuts{end+1}   = cand{c, 1};   %#ok<AGROW>
            baseLabels{end+1} = cand{c, 2};   %#ok<AGROW>
        end
    end

    % labels{1} names the proposed method; labels{k+1} names baseOuts{k}.
    hists  = [{propOut.hist}, cellfun(@(o) o.hist, baseOuts, 'uni', 0)];
    labels = [{propLabel},    baseLabels];

    % ---- figure ----------------------------------------------------------
    if opts.makeFigure
        ttl = sprintf('%s: %s (%s, bias=%s)', datasetName, P.name, ...
                      opts.costMode, opts.biasMode);
        make_config_figure(hists, labels, figPath, ttl, 'Dual suboptimality  f - f*');
        fprintf('Saved %s\n', figPath);
    end

    results          = struct();
    results.skipped  = false;
    results.figPath  = figPath;
    results.problem  = P.name;
    results.costMode = opts.costMode;
    results.biasMode = opts.biasMode;
    results.proposed = propOut;
    results.baseline = baseOuts;     % cell array, aligned with labels(2:end)
    results.labels   = labels;
    results.hists    = hists;
    results.opts     = opts;
end


function report_conditioning(ker)
%REPORT_CONDITIONING  Print sigma_1(K) / max_i K_ii.
%
%   The projected-gradient step is 1/sigma_1(K); the exact coordinate step is
%   1/K_ii. This ratio is therefore the per-coordinate handicap the global
%   Lipschitz constant imposes, and it predicts which family wins.
%
%   Note ker.sig1 carries a 1.05 safety factor; it is divided out here so the
%   printed value is the actual spectral radius.

    maxKii = max(full(diag(ker.K)));             % RBF: identically 1

    sig1 = ker.sig1 / 1.05;
    fprintf('sigma_1(K) = %.4e | max_i K_ii = %.4e | ratio = %.2f\n', ...
            sig1, maxKii, sig1 / maxKii);
end


%% ======================================================================
%  Options
%  ======================================================================

function opts = fill_default_opts(opts)
%FILL_DEFAULT_OPTS  Populate unset fields and validate.

    % problem
    opts = set_default(opts, 'problem',  'l2svm');   % l1svm l2svm svr nusvm mcsvm
    opts = set_default(opts, 'costMode', 'none');
    opts = set_default(opts, 'C',        1);
    opts = set_default(opts, 'nu',       0.2);
    opts = set_default(opts, 'epsSVR',   0.1);

    % kernel (RBF throughout)
    opts = set_default(opts, 'rbfGamma', 2.5);       % k = exp(-g||x - x'||^2),
                                                     % g = 1/(2r^2), radius 1
    opts = set_default(opts, 'explicitKernelMaxN', 80000);

    % bias handling (see the header of run_svm_comparison)
    opts = set_default(opts, 'biasMode', 'constrained');    % 'none' (Track A) |
                                                     % 'constrained' (Track B)

    % proximal solver
    opts = set_default(opts, 'accel',       true);
    opts = set_default(opts, 'lazy',        true);   % incremental K*delta updates
    opts = set_default(opts, 'lazyRefresh', 500);    % full recompute cadence (FP drift)
    opts = set_default(opts, 'maxIters',    5000);
    opts = set_default(opts, 'tol',         1e-12);
    opts = set_default(opts, 'timeLimit',   360);

    % cost-sensitive weighting
    opts = set_default(opts, 'classCosts',    []);   % per-class multipliers
    opts = set_default(opts, 'costPosFactor', 2);    % svr 'cost': over-estimate box
    opts = set_default(opts, 'costNegFactor', 1);    % svr 'cost': under-estimate box

    % task adaptation
    opts = set_default(opts, 'task',         '');
    opts = set_default(opts, 'binarize',     struct('type', 'halfsplit'));
    opts = set_default(opts, 'mcBins',       4);
    opts = set_default(opts, 'maxSamples',   inf);
    opts = set_default(opts, 'standardize',  false);
    opts = set_default(opts, 'standardizeY', false);

    % baselines
    opts = set_default(opts, 'smoTolerances', 10 .^ -(1:8));

    % reporting
    opts = set_default(opts, 'outRoot',    '.');
    opts = set_default(opts, 'overwrite',  false);
    opts = set_default(opts, 'seed',       1);
    opts = set_default(opts, 'verbose',    true);
    opts = set_default(opts, 'printEvery', 200);
    opts = set_default(opts, 'evalEvery',  10);
    opts = set_default(opts, 'makeFigure', true);

    % ---- validation ------------------------------------------------------
    valid = {'l1svm', 'l2svm', 'nusvm', 'mcsvm', 'svr'};
    if ~any(strcmp(opts.problem, valid))
        error('opts.problem must be one of: %s', strjoin(valid, ', '));
    end

    valid = {'none', 'cost'};
    if ~any(strcmp(opts.costMode, valid))
        error('opts.costMode must be ''none'' or ''cost''.');
    end

    valid = {'constrained', 'none'};
    if ~any(strcmp(opts.biasMode, valid))
        error('opts.biasMode must be ''constrained'' or ''none''.');
    end
end


function opts = set_default(opts, name, value)
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = value;
    end
end


%% ======================================================================
%  Data loading and task adaptation
%  ======================================================================

function [X, y] = load_xy_from_mat(matFile)
%LOAD_XY_FROM_MAT  Pull a design matrix and target vector out of a .mat file.
%
%   Tries the usual variable names first, then falls back to any (matrix,
%   vector) pair whose dimensions agree.

    S = load(matFile);

    xCandidates = {'Z', 'X', 'data', 'features', 'A', 'x'};
    yCandidates = {'y', 'Y', 'labels', 'label', 'target', 'targets'};

    X = [];
    y = [];

    for i = 1:numel(xCandidates)
        if isfield(S, xCandidates{i})
            X = S.(xCandidates{i});
            break;
        end
    end

    for i = 1:numel(yCandidates)
        nm = yCandidates{i};
        if ~isfield(S, nm)
            continue;
        end
        yy = S.(nm);
        if (isnumeric(yy) || islogical(yy)) && isvector(yy)
            y = yy;
            break;
        elseif isnumeric(yy) && ismatrix(yy) && size(yy, 1) == size(X, 1) && size(yy, 2) > 1
            [~, y] = max(yy, [], 2);             % one-hot -> labels
            break;
        end
    end

    % Fallback: first (matrix, conforming vector) pair in the file.
    if isempty(X) || isempty(y)
        names = fieldnames(S);
        for i = 1:numel(names)
            A = S.(names{i});
            if (isnumeric(A) || islogical(A)) && ismatrix(A) && ~isvector(A)
                for j = 1:numel(names)
                    b = S.(names{j});
                    if (isnumeric(b) || islogical(b)) && isvector(b) && numel(b) == size(A, 1)
                        X = A;
                        y = b;
                        break;
                    end
                end
            end
            if ~isempty(X) && ~isempty(y)
                break;
            end
        end
    end

    if isempty(X) || isempty(y)
        error(['Could not identify X and y in %s. Store them as X and y, ', ...
               'or extend load_xy_from_mat().'], matFile);
    end

    if ~issparse(X)
        X = double(X);
    end
    y = double(y(:));
end


function task = detect_task(y)
%DETECT_TASK  Guess 'binary', 'multiclass', or 'regression' from the targets.

    isInt = all(abs(y - round(y)) < 1e-9);
    u     = unique(y);

    if isInt && numel(u) <= 2000 && numel(u) <= numel(y) / 2
        if numel(u) == 2
            task = 'binary';
        else
            task = 'multiclass';
        end
    else
        task = 'regression';
    end

    fprintf(['Task not specified; inferred "%s" (%d unique targets). ', ...
             'Override with opts.task.\n'], task, numel(u));
end


function [X, y, meta] = preprocess_xy(X, y, opts)
%PREPROCESS_XY  Orient, subsample, adapt targets to opts.problem, standardize.
%
%   Target adaptation lets any dataset drive any problem, which keeps the
%   benchmark grid rectangular:
%
%       binary problem on multiclass data   -> opts.binarize (default halfsplit)
%       binary problem on regression data   -> median split
%       mcsvm on regression data            -> opts.mcBins quantile bins
%       svr on classification data          -> regress on the +/-1 labels
%
%   The last two are optimization benchmarks, not statements about modelling.

    if size(X, 1) ~= numel(y) && size(X, 2) == numel(y)
        X = X';
    end
    if size(X, 1) ~= numel(y)
        error('X and y dimensions do not match.');
    end

    if isempty(opts.task)
        task = detect_task(y);
    else
        task = opts.task;
    end

    % ---- subsample -------------------------------------------------------
    n = size(X, 1);
    if isfinite(opts.maxSamples) && n > opts.maxSamples
        N   = round(opts.maxSamples);
        sel = randperm(n, N);
        X   = X(sel, :);
        y   = y(sel);
        fprintf('Subsampled %d -> %d rows.\n', n, N);
    end

    % ---- adapt targets to the requested problem --------------------------
    isBinaryProblem = any(strcmp(opts.problem, {'l1svm', 'l2svm', 'nusvm'}));
    K = 1;

    if isBinaryProblem
        switch task
            case 'binary'
                [~, ~, yi] = unique(y);
                y = 2 * (yi == 2) - 1;
            case 'multiclass'
                [~, ~, yi] = unique(y);
                y = binarize_labels(yi, opts.binarize, max(yi));
            case 'regression'
                y = 2 * (y > median(y)) - 1;
                fprintf('Regression targets median-split into +/-1 labels.\n');
        end
        K = 2;

    elseif strcmp(opts.problem, 'mcsvm')
        if strcmp(task, 'regression')
            edges = quantile(y, (1:opts.mcBins - 1) / opts.mcBins);
            yb    = ones(numel(y), 1);
            for e = 1:numel(edges)
                yb = yb + (y > edges(e));
            end
            y = yb;
            fprintf('Regression targets binned into %d classes for mcsvm.\n', opts.mcBins);
        end
        [~, ~, y] = unique(y);
        y = double(y(:));
        K = max(y);
        if K < 2
            error('mcsvm needs at least 2 classes.');
        end

    else  % svr
        if ~strcmp(task, 'regression')
            [~, ~, yi] = unique(y);
            if max(yi) == 2
                y = 2 * (yi == 2) - 1;
            else
                y = binarize_labels(yi, opts.binarize, max(yi));
            end
            fprintf('Classification labels used as +/-1 regression targets for svr.\n');
        elseif opts.standardizeY
            y = (y - mean(y)) / max(std(y), 1e-12);   % makes epsSVR comparable
        end
    end

    % ---- standardize -----------------------------------------------------
    % Sparse input is only scaled, never centred: subtracting the mean would
    % destroy sparsity and blow up memory. NOTE: for already L2-normalized
    % text features (e.g. rcv1) this inflates pairwise distances and collapses
    % the RBF off-diagonals; keep standardize = false there.
    if opts.standardize
        if issparse(X)
            colScale = full(sqrt(sum(X .^ 2, 1) / max(1, size(X, 1))));
            colScale(colScale < 1e-12) = 1;
            X = X * spdiags(1 ./ colScale(:), 0, size(X, 2), size(X, 2));
        else
            mu    = mean(X, 1);
            sigma = std(X, 0, 1);
            sigma(sigma < 1e-12) = 1;
            X = bsxfun(@rdivide, bsxfun(@minus, X, mu), sigma);
        end
    end

    meta = struct('task', task, 'K', K);
end


function y = binarize_labels(y, spec, K)
%BINARIZE_LABELS  Collapse K classes to +/-1.

    switch spec.type
        case 'halfsplit'
            y = 2 * (y <= floor(K / 2)) - 1;
        case 'ovr'
            y = 2 * (y == spec.k) - 1;
        otherwise
            error('Unknown binarize type "%s".', spec.type);
    end

    if all(y == y(1))
        error('Binarization produced a single class; adjust opts.binarize.');
    end
end


%% ======================================================================
%  Problem definition
%  ======================================================================

function P = make_problem(y, meta, opts)
%MAKE_PROBLEM  Boxes, weights, and constraint flags for one SVM variant.
%
%   Fields consumed downstream:
%       P.name       variant
%       P.Ci         per-sample box / diagonal   (l1svm, l2svm, mcsvm)
%       P.up         per-sample box              (nusvm)
%       P.bUp, P.bLo per-sample box              (svr)
%       P.hasEq      does the dual carry a BIAS equality constraint?
%       P.setupGram  setup seconds charged to Gram readers
%       P.setupPG    setup seconds charged to the PG solvers

    n = numel(y);

    P           = struct();
    P.name      = opts.problem;
    P.costMode  = opts.costMode;
    P.C         = opts.C;
    P.nu        = opts.nu;
    P.eps       = opts.epsSVR;
    P.K         = meta.K;
    P.setupGram     = 0;
    P.setupPG       = 0;
    P.setupGramCols = 0;
    P.setupPGCols   = 0;

    withCost = strcmp(opts.costMode, 'cost');

    % P.hasEq is read only by solve_l1l2, solve_svr, baseline_dcd_binary,
    % baseline_dcd_svr and baseline_libsvm_sweep. It means "the dual carries a
    % bias equality that the projection must enforce" -- not "the dual has any
    % equality at all".
    P.hasEq = strcmp(opts.biasMode, 'constrained');
    if any(strcmp(P.name, {'nusvm', 'mcsvm'}))
        % nusvm: two coupled class-mass equalities; they define nu and no bias
        %        reformulation removes them. Its coordinate baseline uses
        %        same-class pairs instead (baseline_pcd_nusvm).
        % mcsvm: sum_j a_ij = 0 is per-ROW, not a bias constraint. No solver
        %        reads hasEq for mcsvm; set here for symmetry only.
        P.hasEq = true;
    end

    switch P.name
        case {'l1svm', 'l2svm'}
            P.Ci = opts.C * class_weights_pm(y, withCost, opts);

        case 'nusvm'
            P.up = class_weights_pm(y, withCost, opts) / n;

            % Eq. 23 needs at least nu/2 of box mass available in each class.
            sp = sum(P.up(y > 0));
            sm = sum(P.up(y < 0));
            if opts.nu / 2 > min(sp, sm) + 1e-12
                error(['nu = %g infeasible: per-class box mass is (%.4g, %.4g) ', ...
                       'but Eq. 23 needs >= nu/2 = %.4g in each. Reduce nu.'], ...
                       opts.nu, sp, sm, opts.nu / 2);
            end

        case 'mcsvm'
            if withCost
                if isempty(opts.classCosts)
                    counts = accumarray(y, 1, [meta.K, 1]);
                    w      = n ./ (meta.K * counts);      % balanced default
                else
                    w = opts.classCosts(:);
                end
                P.Ci = opts.C * w(y);
            else
                P.Ci = opts.C * ones(n, 1);
            end

        case 'svr'
            if withCost
                P.bUp =  opts.C * opts.costPosFactor * ones(n, 1);
                P.bLo = -opts.C * opts.costNegFactor * ones(n, 1);
            else
                P.bUp =  opts.C * ones(n, 1);
                P.bLo = -opts.C * ones(n, 1);
            end
    end
end


function w = class_weights_pm(y, withCost, opts)
%CLASS_WEIGHTS_PM  Per-sample class weights for +/-1 labels.

    n = numel(y);

    if ~withCost
        w = ones(n, 1);
        return;
    end

    if ~isempty(opts.classCosts)
        cc       = opts.classCosts(:);
        w        = cc(1) * ones(n, 1);
        w(y > 0) = cc(2);
    else
        np       = sum(y > 0);
        nm       = n - np;
        w        = ones(n, 1);
        w(y > 0) = n / (2 * np);
        w(y < 0) = n / (2 * nm);
    end
end


%% ======================================================================
%  Kernel operator
%  ======================================================================

function ker = make_kernel_op(X, y, P, opts, pre)
%MAKE_KERNEL_OP  Build the RBF Gram, K*a, and sigma_1(K).
%
%   Returns
%       ker.mul       @(a) -> K*a. Accepts n x 1 and n x K right-hand sides.
%       ker.K         the n x n Gram
%       ker.sig1      1.05 * lambda_max(K)
%       ker.gramTime  seconds spent building K       (shared cost)
%       ker.sig1Time  seconds spent on sigma_1       (PG-only cost)
%
%   PRE (optional) supplies a cached gamma-invariant distance matrix:
%       pre.D2        n x n squared distances from SQ_DISTS
%       pre.d2Time    seconds that build cost
%   When supplied, the Gram is exp(-gamma*D2) and pre.d2Time is ADDED to
%   ker.gramTime. That re-charge matters: gramTime is pre-charged to every
%   solver clock via P.setupGram / P.setupPG, so amortizing the distance
%   build across a sweep must not make the sweep's setup look cheaper than
%   the same run standing alone. Callers that omit PRE are unaffected.
%
%   For l1svm/l2svm/nusvm the Gram is SIGNED: K = (y y') .* Kraw. Callers that
%   index ker.K must account for that.
%
%   The Gram is always materialized: every coordinate baseline needs kernel
%   COLUMNS, and a matrix-free RBF matvec was measured 55-63x slower than
%   dense. n > opts.explicitKernelMaxN is therefore a hard error, raised here,
%   before the 8n^2-byte allocation is attempted.

    if nargin < 5
        pre = [];
    end

    n = size(X, 1);

    if n > opts.explicitKernelMaxN
        error(['make_kernel_op: n = %d > opts.explicitKernelMaxN = %d. The ', ...
               'benchmark requires the cached %d x %d Gram (%.1f GB). ', ...
               'Subsample via opts.maxSamples, or raise the threshold if ', ...
               'memory allows.'], n, opts.explicitKernelMaxN, n, n, 8 * n^2 / 1e9);
    end

    signed = any(strcmp(P.name, {'l1svm', 'l2svm', 'nusvm'}));

    % ---- Gram ------------------------------------------------------------
    tGram    = tic;
    d2Charge = 0;

    if isempty(pre)
        K = rbf_gram(X, X, opts.rbfGamma);
    else
        % Blocked exp so the n x n intermediate (-gamma*D2) is never
        % materialized in full: peak stays at D2 + K + one column block
        % rather than D2 + K + a third n x n array.
        K   = zeros(n, n);
        blk = 2048;
        for j0 = 1:blk:n
            j1 = min(j0 + blk - 1, n);
            K(:, j0:j1) = exp(-opts.rbfGamma * pre.D2(:, j0:j1));
        end
        d2Charge = pre.d2Time;
    end

    if signed
        % Two-pass row/column scaling == (y*y').*K without the n x n dense
        % outer-product temporary (which doubles peak memory at this size).
        K = K .* y;
        K = K .* y.';
    end
    ker.mul      = @(a) K * a;
    ker.explicit = true;
    ker.K        = K;

    ker.gramTime = toc(tGram) + d2Charge;

    % ---- sigma_1 ---------------------------------------------------------
    tSig1 = tic;
    optsE.tol = 1e-8; optsE.issym = true; optsE.isreal = true;
    counted_mul([], ker.mul, 'reset');
    s1  = eigs(@(x) counted_mul(x, ker.mul, 'mul'), n, 1, 'largestabs', optsE);
    nmv = counted_mul([], ker.mul, 'get');
    ker.sig1     = 1.05 * abs(s1);
    ker.sig1Time = toc(tSig1);
    ker.gramCols = n;                             % n columns formed
    ker.sig1Cols = nmv * n;                       % eigs matvecs, n cols each
    ker.setupTime = ker.gramTime + ker.sig1Time;
end


function y = counted_mul(x, mulfun, mode)
%COUNTED_MUL  ker.mul wrapper that counts matvecs (for the eigs setup charge).
%   'reset' zeroes the counter, 'get' returns it, 'mul' applies and counts.
    persistent cnt
    switch mode
        case 'reset', cnt = 0; y = [];
        case 'get',   y = cnt;
        otherwise,    cnt = cnt + 1; y = mulfun(x);
    end
end

function K = rbf_gram(A, B, gamma)
%RBF_GRAM  k(a, b) = exp(-gamma ||a - b||^2).
%
%   gamma = 1/(2 radius^2) matches LIBSVM's -g convention, so radius 1 is
%   gamma 0.5.

    K = exp(-gamma * sq_dists(A, B));
end

function D2 = sq_dists(A, B)
%SQ_DISTS  Pairwise squared Euclidean distances, ||a - b||^2.
%
%   Gamma-invariant, which is the whole reason it is a separate function:
%   gamma_sweep builds this once and reuses it across the sweep, since the
%   only gamma-dependent step is the elementwise exp.
%
%   The negative round-off clamp is applied here, once, rather than on every
%   kernel build.

    sqA = full(sum(A .^ 2, 2));
    sqB = full(sum(B .^ 2, 2));
    D2  = bsxfun(@plus, sqA, sqB') - 2 * full(A * B');
    D2  = max(D2, 0);                             % clamp negative round-off
end

function kacc = dcd_kernel_ops(ker)
%DCD_KERNEL_OPS  Column / diagonal access for the coordinate methods.
%
%   The Gram is always cached (make_kernel_op enforces it), so a coordinate
%   gradient is one column lookup, O(n).
%
%   kacc.diag   n x 1, diag(K)
%   kacc.col    @(i) -> K(:, i)

    kacc.diag = max(full(diag(ker.K)), 1e-12);
    kacc.col  = @(i) ker.K(:, i);
end


%% ======================================================================
%  Objective
%  ======================================================================

function f = plotted_objective(P, a, g, y)
%PLOTTED_OBJECTIVE  Dual objective, MINIMIZED, with g = K*a supplied.
%
%   Every solver maintains g alongside a, so evaluating this is free -- which is
%   what lets the timing clock stay paused during recording. All methods in a
%   given figure minimize this same function, so they share one f*.

    switch P.name
        case 'l1svm'
            f = 0.5 * (a' * g) - sum(a);

        case 'l2svm'
            f = 0.5 * (a' * g) + 0.5 * sum(a .^ 2 ./ P.Ci) - sum(a);

        case 'nusvm'
            f = 0.5 * (a' * g);

        case 'svr'
            f = 0.5 * (a' * g) - y' * a + P.eps * sum(abs(a));

        case 'mcsvm'
            % d(a) = 0.5 <a, K a> - <E, a>, with gradient K*a - E. This is what
            % solve_mcsvm, baseline_smo_mcsvm and baseline_dcd_mcsvm all
            % minimize, so they share d*.
            n       = numel(y);
            idxTrue = sub2ind(size(a), (1:n)', y);
            f       = 0.5 * sum(sum(a .* g)) - sum(a(idxTrue));
    end
end


%% ======================================================================
%  Proposed solvers: accelerated projected gradient on the dual
%  ======================================================================
function out = solve_l1l2(ker, y, P, opts)
%SOLVE_L1L2  Accelerated projected gradient for the L1-/L2-SVM dual.
%
%   Eq. 5:  min_a  0.5 a'Ka + (1/2C)||a||^2 - a'1
%           s.t.   0 <= a <= C_i,  <a, y> = 0
%
%   Iteration, with L = sigma_1(K) (+ 1/C for L2):
%
%       beta <- z - grad/L
%       find lambda with  sum_i y_i clip(beta_i - lambda y_i) = 0   (bisection)
%       a    <- clip(beta - lambda y)
%
%   Under biasMode 'none' the equality is gone, lambda = 0, and the projection
%   collapses to a clip.

    n    = numel(y);
    isL2 = strcmp(P.name, 'l2svm');

    if isL2
        L = ker.sig1 + 1 / min(P.Ci);
    else
        L = ker.sig1;
    end

    alpha        = zeros(n, 1);
    z            = alpha;
    tk           = 1;
    ga           = zeros(n, 1);          % maintained K*alpha  (alpha = 0 -> 0)
    gz           = zeros(n, 1);          % maintained K*z
    sinceRefresh = 0;

    hist = init_hist();
    clk  = clk_new(P.setupPG);
    cols = P.setupPGCols;            % kernel-column units (see header)

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        if isL2
            grad = gz + z ./ P.Ci - 1;
        else
            grad = gz - 1;
        end

        % ---- record (off-clock) ------------------------------------------
        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f    = plotted_objective(P, alpha, ga, y);   % free: ga is maintained
            hist = rec_hist(hist, clk.solve, f, cols);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end

        % ---- gradient step, then project ---------------------------------
        beta  = z - grad / L;
        scale = max(1, max(abs(beta)));

        if P.hasEq
            if isL2
                h = @(lam) sum(y .* max(beta - lam * y, 0));
            else
                h = @(lam) sum(y .* min(P.Ci, max(beta - lam * y, 0)));
            end
            lam = bisect_root(h, scale);
        else
            lam = 0;
        end

        if isL2
            alphaNew = max(beta - lam * y, 0);
        else
            alphaNew = min(P.Ci, max(beta - lam * y, 0));
        end

        dA        = alphaNew - alpha;
        converged = norm(dA) <= opts.tol * max(1, norm(alpha));

        % ---- momentum ----------------------------------------------------
        if isL2
            mu = 1 / max(P.Ci);          % strongly convex
        else
            mu = 0;
        end

        [theta, tk] = momentum_step(opts, mu, L, tk, (z - alphaNew)' * dA);
        z = alphaNew + theta * dA;

        % ---- lazy kernel-image maintenance -------------------------------
        % Bound coordinates have dA_i = 0 exactly (clip to clip), so late-phase
        % deltas are supported on the free SVs only and K*dA is a cheap sparse
        % matvec. z is a linear combination of the last two iterates, so K*z
        % falls out of the two maintained images with no extra matvec.
        sinceRefresh = sinceRefresh + 1;
        gaPrev       = ga;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * n
            ga   = ga + ker.mul(sparse(dA));
            cols = cols + nnz(dA);
        else
            ga           = ker.mul(alphaNew);
            cols         = cols + n;    % periodic full refresh
            sinceRefresh = 0;
        end
        gz = ga + theta * (ga - gaPrev);

        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y), cols);

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end

function out = solve_svr(ker, y, P, opts)
%SOLVE_SVR  Accelerated proximal gradient for the eps-insensitive SVR dual.
%
%   Eq. 10:  min_b  0.5 b'Kb - y'b + eps ||b||_1
%            s.t.   bLo <= b <= bUp,  1'b = 0
%
%   Iteration, with L = sigma_1(K) and S the soft-threshold at eps/L:
%
%       v      <- z - (K z - y)/L
%       lambda <- root of  sum_i clip(S(v_i - lambda)) = 0     (bisection)
%       b      <- clip(S(v - lambda))

    n   = numel(y);
    L   = ker.sig1;
    thr = P.eps / L;
    st  = @(u) sign(u) .* max(abs(u) - thr, 0);   % S_{eps/L}

    b            = zeros(n, 1);
    z            = b;
    tk           = 1;
    gb           = zeros(n, 1);          % maintained K*b
    gz           = zeros(n, 1);          % maintained K*z
    sinceRefresh = 0;

    hist = init_hist();
    clk  = clk_new(P.setupPG);
    cols = P.setupPGCols;            % kernel-column units (see header)

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f    = plotted_objective(P, b, gb, y);
            hist = rec_hist(hist, clk.solve, f, cols);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end

        v = z - (gz - y) / L;

        if P.hasEq
            h   = @(lam) sum(min(P.bUp, max(P.bLo, st(v - lam))));
            lam = bisect_root(h, max(1, max(abs(v))));
        else
            lam = 0;
        end
        bNew = min(P.bUp, max(P.bLo, st(v - lam)));

        dB        = bNew - b;
        converged = norm(dB) <= opts.tol * max(1, norm(b));

        theta = 0;
        if opts.accel
            if (z - bNew)' * dB > 0              % gradient restart
                tk = 1;
            else
                tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
                theta = (tk - 1) / tk1;
                tk    = tk1;
            end
        end
        z = bNew + theta * dB;

        sinceRefresh = sinceRefresh + 1;
        gbPrev       = gb;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dB) < 0.5 * n
            gb   = gb + ker.mul(sparse(dB));
            cols = cols + nnz(dB);
        else
            gb           = ker.mul(bNew);
            cols         = cols + n;
            sinceRefresh = 0;
        end
        gz = gb + theta * (gb - gbPrev);

        b = bNew;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, b, ker.mul(b), y), cols);

    out = struct('alpha', b, 'hist', hist, 'iters', it, 'skipped', false);
end



function out = solve_mcsvm(ker, y, P, opts)
%SOLVE_MCSVM  Jacobi-style parallel projected gradient for Crammer-Singer.
%
%   min_a  0.5 tr(a'Ka) - <a, E>
%   s.t.   0 <= a_{i,y_i} <= C_i,  -C_i <= a_{i,j} <= 0 (j != y_i),  a_i 1 = 0
%
%   Iteration, with L = sigma_1(K):
%
%       B <- Z - (K Z - E)/L
%       for each row i (in parallel): find lambda_i satisfying Eq. 16 by
%       bisection, then a_i <- clip(B_i - lambda_i)
%
%   The row equality is per-row, so all n root-finds are independent and the
%   bisection below runs them simultaneously.

    n = numel(y);
    K = P.K;
    L = ker.sig1;

    E   = full(sparse((1:n)', y, 1, n, K));       % 1_ind
    CiK = repmat(P.Ci, 1, K);
    UP  = E .* CiK;                               % [0, C_i] at the true class
    LO  = (E - 1) .* CiK;                         % [-C_i, 0] elsewhere

    alpha        = zeros(n, K);
    Z            = alpha;
    tk           = 1;
    GA           = zeros(n, K);          % maintained K*alpha
    GZ           = zeros(n, K);          % maintained K*Z
    sinceRefresh = 0;

    hist = init_hist();
    clk  = clk_new(P.setupPG);
    cols = P.setupPGCols;            % kernel-column units (see header)

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f    = plotted_objective(P, alpha, GA, y);   % free: GA is maintained
            hist = rec_hist(hist, clk.solve, f, cols);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end

        B = Z - (GZ - E) / L;

        % ---- Eq. 16 for every row at once --------------------------------
        % Each h_i is nonincreasing in lambda_i, positive at lo and nonpositive
        % at hi, so one vectorized bisection resolves all n roots together.
        lo = min(B, [], 2) - max(P.Ci) - 1;
        hi = max(B, [], 2) + 1;
        for k = 1:80
            mid      = (lo + hi) / 2;
            Cl       = min(UP, max(LO, bsxfun(@minus, B, mid)));
            pos      = sum(Cl, 2) > 0;
            lo(pos)  = mid(pos);
            hi(~pos) = mid(~pos);
        end
        mid      = (lo + hi) / 2;
        alphaNew = min(UP, max(LO, bsxfun(@minus, B, mid)));

        dA        = alphaNew - alpha;
        converged = norm(dA(:)) <= opts.tol * max(1, norm(alpha(:)));

        theta = 0;
        if opts.accel
            if sum(sum((Z - alphaNew) .* dA)) > 0    % gradient restart
                tk = 1;
            else
                tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
                theta = (tk - 1) / tk1;
                tk    = tk1;
            end
        end
        Z = alphaNew + theta * dA;

        % Rows whose every coordinate stayed clipped at its bound have an
        % all-zero delta row, so K*dA touches only the active rows.
        sinceRefresh = sinceRefresh + 1;
        GAPrev       = GA;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * numel(dA)
            GA   = GA + ker.mul(sparse(dA));
            cols = cols + nnz(dA);
        else
            GA           = ker.mul(alphaNew);
            cols         = cols + n * K;
            sinceRefresh = 0;
        end
        GZ = GA + theta * (GA - GAPrev);

        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y), cols);

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end

function out = solve_nusvm(ker, y, P, opts)
%SOLVE_NUSVM  Accelerated projected gradient for the nu-SVM dual.
%
%   min_a  0.5 a'Ka
%   s.t.   0 <= a <= up,  <a, y> = 0,  sum(a) >= nu
%
%   There is no linear term, so the mass constraint is what makes the problem
%   non-trivial: drop it and a = 0 is optimal. Each iteration projects onto
%   {0 <= a <= up, y'a = 0} (Eq. 21) and, if the mass constraint is violated,
%   re-projects with one multiplier per class so that each class carries nu/2
%   (Eqs. 22-23).

    n  = numel(y);
    L  = ker.sig1;
    up = P.up;
    ip = (y > 0);
    im = ~ip;

    alpha        = zeros(n, 1);
    z            = alpha;
    tk           = 1;
    ga           = zeros(n, 1);          % maintained K*alpha
    gz           = zeros(n, 1);          % maintained K*z
    sinceRefresh = 0;

    hist = init_hist();
    clk  = clk_new(P.setupPG);
    cols = P.setupPGCols;            % kernel-column units (see header)

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        % alpha = 0 at it == 1 violates sum(alpha) >= nu and has f = 0 < f*,
        % which would wreck a log-suboptimality plot. Start recording once the
        % iterate is feasible, i.e. after the first projection.
        if it > 1 && mod(it - 1, opts.evalEvery) == 0
            f    = plotted_objective(P, alpha, ga, y);
            hist = rec_hist(hist, clk.solve, f, cols);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end

        B = z - gz / L;

        % ---- Eq. 21: project onto {0 <= a <= up, y'a = 0} -----------------
        h   = @(lam) sum(y .* min(up, max(B - lam * y, 0)));
        lam = bisect_root(h, max(1, max(abs(B))));
        a   = min(up, max(B - lam * y, 0));

        % ---- Eqs. 22-23: enforce nu/2 of mass in each class ---------------
        if sum(a) < P.nu - 1e-12
            hp = @(l) sum(min(up(ip), max(B(ip) - l, 0))) - P.nu / 2;
            hm = @(l) sum(min(up(im), max(B(im) - l, 0))) - P.nu / 2;
            lp = bisect_root(hp, max(1, max(abs(B(ip)))));
            lm = bisect_root(hm, max(1, max(abs(B(im)))));

            a(ip) = min(up(ip), max(B(ip) - lp, 0));
            a(im) = min(up(im), max(B(im) - lm, 0));
        end

        dA        = a - alpha;
        converged = norm(dA) <= opts.tol * max(1, norm(alpha));

        theta = 0;
        if opts.accel
            if (z - a)' * dA > 0                 % gradient restart
                tk = 1;
            else
                tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
                theta = (tk - 1) / tk1;
                tk    = tk1;
            end
        end
        z = a + theta * dA;

        sinceRefresh = sinceRefresh + 1;
        gaPrev       = ga;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * n
            ga   = ga + ker.mul(sparse(dA));
            cols = cols + nnz(dA);
        else
            ga           = ker.mul(a);
            cols         = cols + n;
            sinceRefresh = 0;
        end
        gz = ga + theta * (ga - gaPrev);

        alpha = a;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y), cols);

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end



%% ======================================================================
%  Baselines: SMO
%  ======================================================================

function out = baseline_libsvm_sweep(ker, X, y, P, opts)
%BASELINE_LIBSVM_SWEEP  LIBSVM at a sequence of tightening tolerances.
%
%   LIBSVM is a black box: it exposes no per-iteration hook, so a trajectory is
%   traced by re-training at each tolerance in opts.smoTolerances and recording
%   (training time, dual objective) for each.
%
%   Timing. With -t 2 LIBSVM builds its own kernel cache inside svmtrain, so
%   the recorded time is exactly what it spends and nothing is pre-charged.
%   With -t 4 (l2svm) it needs a precomputed Gram that we must build for it;
%   that build is timed and added to every recorded point, since LIBSVM cannot
%   run without it.
%
%   Variant mapping:
%       l1svm  -s 0            same dual; -wi carries the class costs
%       l2svm  -s 0 -t 4       precomputed K + diag(1./C_i) with a huge box --
%                              the classic reduction of L2-SVM to hard-margin
%                              SMO. Needs the explicit Gram.
%       nusvm  -s 1            alphas rescaled so sum = nu; see below
%       svr    -s 3

    hist = init_hist();
    out  = struct('alpha', [], 'hist', hist, 'skipped', true);

    if ~P.hasEq && any(strcmp(P.name, {'l1svm', 'l2svm', 'svr'}))
        warning(['baseline_libsvm_sweep: biasMode = ''%s'' puts PG/DCD on the ' ...
                'box-only dual, but LIBSVM always enforces <alpha,y> = 0. ' ...
                'f*_eq >= f*_box, so the curves are not comparable. Skipping. ' ...
                'Use biasMode = ''constrained'' for the matched track.'], opts.biasMode);
        return;                                  % out.skipped is already true
    end

    if exist('svmtrain', 'file') ~= 3
        warning('LIBSVM mex (svmtrain) not on path; skipping SMO baseline.');
        return;
    end

    n       = size(X, 1);
    useKtil = false;
    kpart   = sprintf('-t 2 -g %.10g', opts.rbfGamma);   % matched gamma

    switch P.name
        case 'l1svm'
            base = sprintf('-s 0 %s -c %.10g', kpart, P.C);
            if strcmp(P.costMode, 'cost')
                wp   = P.Ci(find(y > 0, 1)) / P.C;
                wm   = P.Ci(find(y < 0, 1)) / P.C;
                base = sprintf('%s -w1 %.10g -w-1 %.10g', base, wp, wm);
            end

        case 'l2svm'
            base    = sprintf('-s 0 -t 4 -c %.10g', 1e10 * max(P.Ci));
            useKtil = true;

        case 'nusvm'
            if strcmp(P.costMode, 'cost')
                warning('Stock LIBSVM has no weighted nu-SVC; skipping baseline.');
                return;
            end
            base = sprintf('-s 1 %s -n %.10g', kpart, P.nu);

        case 'svr'
            if strcmp(P.costMode, 'cost')
                warning(['Stock LIBSVM eps-SVR has a single C (no asymmetric box); ', ...
                         'skipping baseline.']);
                return;
            end
            base = sprintf('-s 3 %s -c %.10g -p %.10g', kpart, P.C, P.eps);

        otherwise
            return;
    end

    % ---- kernel preparation (charged only when LIBSVM cannot do it) -------
    setupCost = 0;
    if useKtil
        tKtr      = tic;
        Ktr       = [(1:n)', rbf_gram(X, X, opts.rbfGamma) + diag(1 ./ P.Ci)];
        setupCost = toc(tKtr);
    else
        Xsp = sparse(X);                          % format conversion only
    end

    out.skipped = false;

    % ---- starting point --------------------------------------------------
    z0 = zeros(n, 1);
    if strcmp(P.name, 'nusvm')
        ip = (y > 0);
        im = ~ip;
        hp = @(l) sum(min(P.up(ip), max(-l, 0))) - P.nu / 2;
        hm = @(l) sum(min(P.up(im), max(-l, 0))) - P.nu / 2;
        lp = bisect_root(hp, 1);
        lm = bisect_root(hm, 1);

        z0(ip) = min(P.up(ip), max(-lp, 0));
        z0(im) = min(P.up(im), max(-lm, 0));
        hist   = rec_hist(hist, setupCost, plotted_objective(P, z0, ker.mul(z0), y));
    else
        hist = rec_hist(hist, setupCost, plotted_objective(P, z0, zeros(n, 1), y));
    end

    % ---- tolerance sweep -------------------------------------------------
    tolList = opts.smoTolerances(:)';
    a       = zeros(n, 1);

    for t = 1:numel(tolList)
        args = sprintf('%s -e %.3g -q', base, tolList(t));

        tTrain = tic;
        if useKtil
            model = svmtrain(y, Ktr, args);      %#ok<SVMTRAIN>
        else
            model = svmtrain(y, Xsp, args);      %#ok<SVMTRAIN>
        end
        tTrain = setupCost + toc(tTrain);

        a = zeros(n, 1);
        switch P.name
            case {'l1svm', 'l2svm'}
                a(model.sv_indices) = abs(model.sv_coef);   % |y_i a_i| = a_i

            case 'nusvm'
                % LIBSVM rescales nu-SVC alphas internally. The direction is
                % exact, so restore the scale via the active constraint
                % sum(alpha) = nu, then repair the residual constraint error by
                % projecting onto {0 <= a <= up, per-class mass = nu/2} -- two
                % bisections, the same structure as Eq. 23. Without the repair
                % the small violations surface as a fake accuracy floor for the
                % baseline.
                a(model.sv_indices) = abs(model.sv_coef);
                s = sum(a);
                if s > 0
                    a = a * (P.nu / s);
                end

                ip = (y > 0);
                im = ~ip;
                hp = @(l) sum(min(P.up(ip), max(a(ip) - l, 0))) - P.nu / 2;
                hm = @(l) sum(min(P.up(im), max(a(im) - l, 0))) - P.nu / 2;
                lp = bisect_root(hp, max(1, max(abs(a))));
                lm = bisect_root(hm, max(1, max(abs(a))));

                a(ip) = min(P.up(ip), max(a(ip) - lp, 0));
                a(im) = min(P.up(im), max(a(im) - lm, 0));

            case 'svr'
                a(model.sv_indices) = model.sv_coef;        % beta = alpha - alpha*
        end

        f    = plotted_objective(P, a, ker.mul(a), y);
        hist = rec_hist(hist, tTrain, f);
        maybe_print(opts, 'smo-libsvm', t, f);

        if tTrain >= opts.timeLimit
            break;                               % tighter tolerances only cost more
        end
    end

    out.alpha = a;
    out.hist  = hist;
end


function out = baseline_smo_mcsvm(ker, X, y, P, opts)   %#ok<INUSL>
%BASELINE_SMO_MCSVM  Kernelized Crammer-Singer SMO.
%
%   Maximal-violating-pair working set (two classes within one example) and an
%   analytic two-variable step. The pair lies inside a single row, so the row
%   equality sum_j a_ij = 0 is preserved by construction.
%
%   Note this is FIRST-order working-set selection. LIBSVM's default is
%   second-order and is meaningfully faster; keep that in mind when reading the
%   margin over this baseline.

    hist = init_hist();
    out  = struct('alpha', [], 'hist', hist, 'skipped', false);

    n = numel(y);
    K = P.K;

    E   = full(sparse((1:n)', y, 1, n, K));
    CiK = repmat(P.Ci, 1, K);
    UP  = E .* CiK;                               % [0, C_i] at the true class
    LO  = (E - 1) .* CiK;                         % [-C_i, 0] elsewhere

    Kdiag = max(full(diag(ker.K)), 1e-12);

    alpha = zeros(n, K);
    KA    = zeros(n, K);                          % maintained K*alpha (= scores)
    epsB  = 1e-12;                                % box-activity tolerance

    clk  = clk_new(P.setupGram);                  % reads ker.K columns
    cols = P.setupGramCols;
    hist = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, KA, y), cols);

    chkEvery = max(1, n);                         % record cadence ~ one epoch
    maxSteps = opts.maxIters * max(1, n);

    s = 0;
    while s < maxSteps
        s = s + 1;

        % ---- maximal violating pair, vectorized over all rows -------------
        G   = KA - E;                             % gradient K*alpha - E
        Gup = G;  Gup(alpha >= UP - epsB) = +inf; % classes that can rise
        Gdn = G;  Gdn(alpha <= LO + epsB) = -inf; % classes that can fall

        [GminUp, uIdx] = min(Gup, [], 2);
        [GmaxDn, vIdx] = max(Gdn, [], 2);

        viol       = GmaxDn - GminUp;             % KKT violation per row
        [mviol, i] = max(viol);
        if ~(mviol > opts.tol)                    % KKT-satisfied to tol
            break;
        end

        % ---- analytic two-variable step in row i: a_u += t, a_v -= t ------
        u = uIdx(i);
        v = vIdx(i);

        t = mviol / (2 * Kdiag(i));                                   % free minimum
        t = min([t, UP(i, u) - alpha(i, u), alpha(i, v) - LO(i, v)]); % clip to box

        alpha(i, u) = alpha(i, u) + t;
        alpha(i, v) = alpha(i, v) - t;

        Ki = ker.K(:, i);
        KA(:, u) = KA(:, u) + t * Ki;             % only two columns move
        KA(:, v) = KA(:, v) - t * Ki;
        cols     = cols + 2;                      % one column read, 2-wide use

        % ---- record / time gate ------------------------------------------
        if mod(s, chkEvery) == 0
            clk  = clk_pause(clk);
            f    = plotted_objective(P, alpha, KA, y);
            hist = rec_hist(hist, clk.solve, f, cols);
            maybe_print(opts, 'smo-cs', s, f);
            stop = clk.solve >= opts.timeLimit;
            clk  = clk_resume(clk);
            if stop
                break;
            end
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, KA, y), cols);

    out.alpha = alpha;
    out.hist  = hist;
end


%% ======================================================================
%  Baselines: dual coordinate descent
%
%  All four minimize the same plotted_objective as the PG solvers, and all four
%  replace the GLOBAL step 1/sigma_1(K) with the LOCAL exact step 1/K_ii (or
%  1/eta for the nu-SVM pair step). That substitution is the whole comparison:
%  the ratio sigma_1(K) / max_i K_ii is the handicap the global Lipschitz
%  constant imposes, and it predicts which family wins.
%
%  Cost is matched by construction. One coordinate update is O(n) with a cached
%  Gram column, so an epoch of n updates is O(n^2) -- exactly one PG matvec.
%
%  BLOCK SIZE is dictated by the coupling in each dual:
%      l1/l2/svr (bias-free)  1 coordinate         box only
%      mcsvm                  1 row, K coords      equality is per-row
%      nusvm                  2 coords, same class two class-mass equalities
%  ======================================================================

function out = baseline_dcd_binary(ker, X, y, P, opts)   %#ok<INUSL>
%BASELINE_DCD_BINARY  Kernel SOR / kernel-adatron for L1-/L2-SVM.
%
%   Mangasarian & Musicant (1999); Friess et al. (1998). Equivalently, Hsieh et
%   al. (2008) Algorithm 3 -- random permutation plus shrinking -- with a kernel
%   gradient (a cached Gram column at O(n)) in place of the linear w-trick.
%
%   Dual (box-only; requires biasMode 'none'):
%       l1svm:  f(a) = 0.5 a'Ka - 1'a                      0 <= a_i <= C_i
%       l2svm:  f(a) = 0.5 a'Ka + 0.5 sum(a^2/C_i) - 1'a   a_i >= 0
%
%   Coordinate i, with g = K*a maintained:
%       G       = g_i - 1                (l1)  |  g_i + a_i/C_i - 1  (l2)
%       Qbar_ii = K_ii                   (l1)  |  K_ii + 1/C_i       (l2)
%       a_i    <- clip(a_i - G/Qbar_ii, 0, U_i),  U_i = C_i (l1) or Inf (l2)
%
%   On RBF, K_ii = 1 identically, so Qbar_ii is 1 or 1 + 1/C_i. No sigma_1
%   appears anywhere -- that is the point.

    hist = init_hist();
    out  = struct('alpha', [], 'hist', hist, 'skipped', true);

    if ~any(strcmp(P.name, {'l1svm', 'l2svm'}))
        return;
    end

    if ~isfield(P, 'hasEq') || P.hasEq
        warning(['baseline_dcd_binary: the dual still carries <alpha,y> = 0, on ', ...
                 'which a single-coordinate move is infeasible. Set opts.biasMode ', ...
                 'to ''none'' to put both solvers on the box-only dual. Skipping.']);
        return;
    end

    kacc = dcd_kernel_ops(ker);

    n    = numel(y);
    isL2 = strcmp(P.name, 'l2svm');

    if isL2
        U    = inf(n, 1);
        Qbar = kacc.diag + 1 ./ P.Ci;
    else
        U    = P.Ci;
        Qbar = kacc.diag;
    end

    alpha = zeros(n, 1);
    g     = zeros(n, 1);                 % g = K*alpha  (alpha = 0 -> 0)

    % Shrinking state (Hsieh et al. Alg. 3).
    A    = (1:n)';
    Mbar =  inf;
    mbar = -inf;

    clk      = clk_new(P.setupGram);
    cols     = P.setupGramCols;
    hist     = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, g, y), cols);
    chkEvery = max(1, ceil(n / 8));      % sub-epoch recording; epochs are long

    ep     = 0;
    steps  = 0;
    timeUp = false;

    while ep < opts.maxIters
        ep = ep + 1;

        M = -inf;
        m =  inf;

        A    = A(randperm(numel(A)));    % Sec. 3.1: random permutation
        keep = true(numel(A), 1);

        for s = 1:numel(A)
            i     = A(s);
            steps = steps + 1;

            if isL2
                G = g(i) + alpha(i) / P.Ci(i) - 1;
            else
                G = g(i) - 1;
            end

            % ---- shrink test, then projected gradient ---------------------
            if alpha(i) <= 0
                if G > Mbar
                    keep(s) = false;     % Thm. 2.1: stays at 0
                    continue;
                end
                PG = min(G, 0);

            elseif alpha(i) >= U(i)
                if G < mbar
                    keep(s) = false;     % Thm. 2.2: stays at U
                    continue;
                end
                PG = max(G, 0);

            else
                PG = G;
            end

            M = max(M, PG);
            m = min(m, PG);

            % ---- exact one-dimensional minimizer -------------------------
            if PG ~= 0
                aOld     = alpha(i);
                alpha(i) = min(max(aOld - G / Qbar(i), 0), U(i));
                delta    = alpha(i) - aOld;

                if delta ~= 0
                    g    = g + delta * kacc.col(i);       % O(n), cached column
                    cols = cols + 1;
                end
            end

            % ---- record / time gate --------------------------------------
            if mod(steps, chkEvery) == 0
                clk  = clk_pause(clk);
                f    = plotted_objective(P, alpha, g, y);
                hist = rec_hist(hist, clk.solve, f, cols);
                maybe_print(opts, 'dcd', steps, f);

                timeUp = clk.solve >= opts.timeLimit;
                clk    = clk_resume(clk);
                if timeUp
                    break;               % A = A(keep) below handles the mask
                end
            end
        end

        A = A(keep);                     % applied exactly once per epoch
        if timeUp
            break;
        end

        % ---- stop / un-shrink (Alg. 3, step 3) ---------------------------
        % Per the Oct-2020 footnote to Hsieh et al., M - m <= tol is not safe on
        % its own: at alpha = 0 every grad_i = -1, so M = m = -1 passes a gap
        % test while being nowhere near optimal. |M| and |m| are checked too.
        if (M - m <= opts.tol) && (abs(M) <= opts.tol) && (abs(m) <= opts.tol)
            if numel(A) == n
                break;
            end
            A    = (1:n)';               % reactivate everything and re-verify
            Mbar =  inf;
            mbar = -inf;
            continue;
        end

        if M <= 0, Mbar =  inf; else, Mbar = M; end
        if m >= 0, mbar = -inf; else, mbar = m; end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, g, y), cols);

    out.alpha   = alpha;
    out.hist    = hist;
    out.skipped = false;
end


function out = baseline_dcd_svr(ker, X, y, P, opts)   %#ok<INUSL>
%BASELINE_DCD_SVR  Coordinate descent for the eps-insensitive SVR dual.
%
%   Kernel form of Ho & Lin (2012) / LIBLINEAR -s 11..13.
%
%   Dual (box-only; requires biasMode 'none'):
%       f(b) = 0.5 b'Kb - y'b + eps ||b||_1,   bLo_i <= b_i <= bUp_i
%
%   The one-variable subproblem picks up the l1 term:
%       min_t  0.5 K_ii t^2 + c t + eps |t|,   c = g_i - K_ii b_i - y_i
%   whose exact minimizer is soft-threshold, THEN clip:
%       b_i <- clip( S_{eps/K_ii}( b_i - (g_i - y_i)/K_ii ), bLo_i, bUp_i )
%
%   Structurally identical to solve_svr's prox step, with the local curvature
%   K_ii in place of the global 1/sigma_1(K).

    hist = init_hist();
    out  = struct('alpha', [], 'hist', hist, 'skipped', true);

    if ~strcmp(P.name, 'svr')
        return;
    end

    if ~isfield(P, 'hasEq') || P.hasEq
        warning(['baseline_dcd_svr: the dual still carries 1''beta = 0. Set ', ...
                 'opts.biasMode to ''none'' to compare on the box-only dual ', ...
                 'that coordinate descent actually solves. Skipping.']);
        return;
    end

    kacc = dcd_kernel_ops(ker);

    n      = numel(y);
    epsIns = P.eps;
    Kii    = kacc.diag;

    b = zeros(n, 1);
    g = zeros(n, 1);                     % g = K*b

    A    = (1:n)';
    Mbar =  inf;
    mbar = -inf;

    clk      = clk_new(P.setupGram);
    cols     = P.setupGramCols;
    hist     = rec_hist(hist, P.setupGram, plotted_objective(P, b, g, y), cols);
    chkEvery = max(1, ceil(n / 8));

    ep     = 0;
    steps  = 0;
    timeUp = false;

    while ep < opts.maxIters
        ep = ep + 1;

        M = -inf;
        m =  inf;

        A    = A(randperm(numel(A)));
        keep = true(numel(A), 1);

        for s = 1:numel(A)
            i     = A(s);
            steps = steps + 1;

            G  = g(i) - y(i);            % smooth part of the gradient
            bi = b(i);

            % ---- projected gradient of the NONSMOOTH objective ------------
            % The eps|b_i| kink makes the subdifferential at b_i = 0 the
            % interval [G - eps, G + eps]; b_i = 0 is optimal iff |G| <= eps.
            if bi >= P.bUp(i)
                PG = max(G + epsIns, 0);
                if G + epsIns < mbar
                    keep(s) = false;
                    continue;
                end

            elseif bi <= P.bLo(i)
                PG = min(G - epsIns, 0);
                if G - epsIns > Mbar
                    keep(s) = false;
                    continue;
                end

            elseif bi > 0
                PG = G + epsIns;

            elseif bi < 0
                PG = G - epsIns;

            else
                PG = min(G + epsIns, 0) + max(G - epsIns, 0);   % 0 iff |G| <= eps
            end

            M = max(M, PG);
            m = min(m, PG);

            % ---- soft-threshold, then clip -------------------------------
            if PG ~= 0
                u     = bi - G / Kii(i);
                thr   = epsIns / Kii(i);
                bNew  = sign(u) * max(abs(u) - thr, 0);          % S_{eps/K_ii}
                bNew  = min(P.bUp(i), max(P.bLo(i), bNew));
                delta = bNew - bi;

                if delta ~= 0
                    b(i) = bNew;
                    g    = g + delta * kacc.col(i);
                    cols = cols + 1;
                end
            end

            if mod(steps, chkEvery) == 0
                clk  = clk_pause(clk);
                f    = plotted_objective(P, b, g, y);
                hist = rec_hist(hist, clk.solve, f, cols);
                maybe_print(opts, 'dcd-svr', steps, f);

                timeUp = clk.solve >= opts.timeLimit;
                clk    = clk_resume(clk);
                if timeUp
                    break;
                end
            end
        end

        A = A(keep);                     % applied exactly once per epoch
        if timeUp
            break;
        end

        if (M - m <= opts.tol) && (abs(M) <= opts.tol) && (abs(m) <= opts.tol)
            if numel(A) == n
                break;
            end
            A    = (1:n)';
            Mbar =  inf;
            mbar = -inf;
            continue;
        end

        if M <= 0, Mbar =  inf; else, Mbar = M; end
        if m >= 0, mbar = -inf; else, mbar = m; end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, b, g, y), cols);

    out.alpha   = b;
    out.hist    = hist;
    out.skipped = false;
end


function out = baseline_dcd_mcsvm(ker, X, y, P, opts)   %#ok<INUSL>
%BASELINE_DCD_MCSVM  Crammer-Singer sequential dual (Keerthi et al. 2008).
%
%   Needs no bias reformulation: Crammer-Singer has no bias term, and its
%   equality sum_j a_ij = 0 is per-ROW, so exact block-coordinate descent over
%   rows is feasible as-is.
%
%   Block minimizer for row i, with g_i = (K a)_{i,:}:
%
%       a_i <- Proj_{Omega_i}( a_i - (g_i - E_i) / K_ii )
%
%   against solve_mcsvm's Jacobi step:
%
%       a_i <- Proj_{Omega_i}( a_i - (g_i - E_i) / sigma_1(K) )
%
%   Same projection, same bisection. The only differences are the step -- local
%   curvature K_ii versus the global Lipschitz constant -- and Gauss-Seidel
%   (row i sees rows 1..i-1 from this epoch) rather than Jacobi.
%
%   Shrinking is exact here: the projection IS the block minimizer, so a zero
%   delta certifies the row optimal given the others.

    hist = init_hist();
    out  = struct('alpha', [], 'hist', hist, 'skipped', true);

    if ~strcmp(P.name, 'mcsvm')
        return;
    end

    kacc = dcd_kernel_ops(ker);

    n = numel(y);
    K = P.K;

    E   = full(sparse((1:n)', y, 1, n, K));       % 1_ind
    CiK = repmat(P.Ci, 1, K);
    UP  = E .* CiK;                               % [0, C_i] at the true class
    LO  = (E - 1) .* CiK;                         % [-C_i, 0] elsewhere

    Kii = kacc.diag;

    alpha = zeros(n, K);
    KA    = zeros(n, K);                          % maintained K*alpha

    A = (1:n)';                                   % active rows

    clk      = clk_new(P.setupGram);
    cols     = P.setupGramCols;
    hist     = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, KA, y), cols);
    chkEvery = max(1, ceil(n / 8));

    ep     = 0;
    steps  = 0;
    timeUp = false;

    while ep < opts.maxIters
        ep = ep + 1;

        maxViol = 0;
        A       = A(randperm(numel(A)));
        keep    = true(numel(A), 1);

        for s = 1:numel(A)
            i     = A(s);
            steps = steps + 1;

            gi = KA(i, :);

            % Exact block minimizer: the capped-simplex projection, stepped by
            % 1/K_ii rather than 1/sigma_1(K).
            B    = alpha(i, :) - (gi - E(i, :)) / Kii(i);
            aNew = cs_row_project(B, LO(i, :), UP(i, :), P.Ci(i));

            dRow = aNew - alpha(i, :);
            viol = max(abs(dRow));

            if viol <= opts.tol
                keep(s) = false;                  % row is optimal; shrink it
            else
                maxViol     = max(maxViol, viol);
                alpha(i, :) = aNew;
                KA          = KA + kacc.col(i) * dRow;   % O(nK)
                cols        = cols + K;          % one column against a 1 x K RHS
            end

            if mod(steps, chkEvery) == 0
                clk  = clk_pause(clk);
                f    = plotted_objective(P, alpha, KA, y);
                hist = rec_hist(hist, clk.solve, f, cols);
                maybe_print(opts, 'dcd-cs', steps, f);

                timeUp = clk.solve >= opts.timeLimit;
                clk    = clk_resume(clk);
                if timeUp
                    break;
                end
            end
        end

        A = A(keep);                              % applied exactly once per epoch
        if timeUp
            break;
        end

        if maxViol <= opts.tol
            if numel(A) == n
                break;                            % optimal over all rows
            end
            A = (1:n)';                           % reactivate and re-verify
            continue;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, KA, y), cols);

    out.alpha   = alpha;
    out.hist    = hist;
    out.skipped = false;
end


function out = baseline_pcd_nusvm(ker, X, y, P, opts)   %#ok<INUSL>
%BASELINE_PCD_NUSVM  Pairwise (same-class) dual coordinate descent for nu-SVM.
%
%   WHY PAIRS. The nu-SVM dual is
%
%       min  0.5 a'Ka                     <- no linear term
%       s.t. 0 <= a_i <= u_i
%            sum_{y_i=+1} a_i = nu/2
%            sum_{y_i=-1} a_i = nu/2
%
%   Two coupled equalities, and they are NOT bias constraints -- they define nu.
%   Drop them and a = 0 is optimal, because there is nothing to push against. So
%   unlike l1/l2/svr, no reformulation makes a single-coordinate move feasible:
%   changing one a_i breaks its class sum.
%
%   The smallest feasible move is a same-class PAIR: a_i += t, a_j -= t with
%   y_i = y_j. Both class sums are preserved, and so is <a, y> = 0. Along
%   d = e_i - e_j,
%
%       f(a + t d) = f(a) + t (g_i - g_j) + 0.5 t^2 (K_ii - 2 K_ij + K_jj)
%
%   so the exact minimizer is t* = -(g_i - g_j)/eta, clipped to the box. This is
%   precisely LIBSVM's nu-SVC working-set rule -- same-class pairs are why
%   nu-SVC needs a different WSS from C-SVC.
%
%   K is SIGNED here. Within a class y_i y_j = +1, so K_ij = Kraw_ij and, on
%   RBF, eta = 2(1 - Kraw_ij) >= 0, zero only for duplicate points.
%
%   Working-set selection is first-order (maximal violating pair within each
%   class). LIBSVM's is second-order and faster; this baseline exists for the
%   instrumented trajectory, not to be the strongest possible competitor.

    hist = init_hist();
    out  = struct('alpha', [], 'hist', hist, 'skipped', true);

    if ~strcmp(P.name, 'nusvm')
        return;
    end

    n  = numel(y);
    up = P.up;
    ip = (y > 0);
    im = ~ip;

    Kdiag = max(full(diag(ker.K)), 1e-12);

    % ---- feasible start --------------------------------------------------
    % Pair updates PRESERVE the class sums; they cannot establish them. So the
    % iterate must start feasible: fill nu/2 of mass into each class, spread
    % evenly and capped at u_i. Same construction baseline_libsvm_sweep uses.
    alpha = zeros(n, 1);
    hp = @(l) sum(min(up(ip), max(-l, 0))) - P.nu / 2;
    hm = @(l) sum(min(up(im), max(-l, 0))) - P.nu / 2;
    lp = bisect_root(hp, 1);
    lm = bisect_root(hm, 1);

    alpha(ip) = min(up(ip), max(-lp, 0));
    alpha(im) = min(up(im), max(-lm, 0));

    clk  = clk_new(P.setupGram);
    g    = ker.mul(alpha);                        % g = K*alpha, maintained below
    cols = P.setupGramCols + n;                   % feasible-start image
    hist = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, g, y), cols);

    epsB     = 1e-12;                             % box-activity tolerance
    chkEvery = max(1, n);                         % record cadence ~ one epoch
    maxSteps = opts.maxIters * max(1, n);

    idxP = find(ip);
    idxM = find(im);

    s = 0;
    while s < maxSteps
        s = s + 1;

        % ---- maximal violating pair, computed WITHIN each class -----------
        % KKT: at the optimum there is a multiplier rho_c per class c with
        %     0 < a_i < u_i  =>  g_i = rho_c
        %     a_i = 0        =>  g_i >= rho_c
        %     a_i = u_i      =>  g_i <= rho_c
        % so a violation exists iff, within a class,
        %     max{g_j : a_j > 0} > min{g_i : a_i < u_i}.
        % Moving t > 0 along e_i - e_j changes f at rate g_i - g_j, so take i
        % with SMALL g (room to rise) and j with LARGE g (room to fall).
        gUp = g;  gUp(alpha >= up - epsB) = +inf; % can increase
        gDn = g;  gDn(alpha <= epsB)      = -inf; % can decrease

        viol = -inf;
        i    = 0;
        j    = 0;

        for c = 1:2
            if c == 1
                idx = idxP;
            else
                idx = idxM;
            end

            [gi, ai] = min(gUp(idx));
            [gj, aj] = max(gDn(idx));

            if gj - gi > viol
                viol = gj - gi;
                i    = idx(ai);                   % rises
                j    = idx(aj);                   % falls
            end
        end

        if ~(viol > opts.tol) || i == 0 || i == j
            break;                                % KKT-satisfied to tol
        end

        % ---- exact two-variable step: a_i += t, a_j -= t ------------------
        Ki  = ker.K(:, i);
        Kj  = ker.K(:, j);
        eta = max(Kdiag(i) - 2 * Ki(j) + Kdiag(j), 1e-12);

        t   = (g(j) - g(i)) / eta;                % = -(g_i - g_j)/eta
        tLo = max(-alpha(i), alpha(j) - up(j));   % 0 <= a_i + t,  a_j - t <= u_j
        tHi = min(up(i) - alpha(i), alpha(j));    % a_i + t <= u_i,  0 <= a_j - t
        t   = min(max(t, tLo), tHi);

        if t == 0
            break;                                % no feasible improving move
        end

        alpha(i) = alpha(i) + t;
        alpha(j) = alpha(j) - t;
        g        = g + t * (Ki - Kj);             % O(n)
        cols     = cols + 2;                      % two column reads

        % ---- record / time gate ------------------------------------------
        if mod(s, chkEvery) == 0
            clk  = clk_pause(clk);
            f    = plotted_objective(P, alpha, g, y);
            hist = rec_hist(hist, clk.solve, f, cols);
            maybe_print(opts, 'pcd-nu', s, f);
            stop = clk.solve >= opts.timeLimit;
            clk  = clk_resume(clk);
            if stop
                break;
            end
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, g, y), cols);

    out.alpha   = alpha;
    out.hist    = hist;
    out.skipped = false;
end


function a = cs_row_project(B, lo, up, Ci)
%CS_ROW_PROJECT  Project one row onto {lo <= a <= up, sum(a) = 0}.
%
%   a = clip(B - lam, lo, up), where h(lam) = sum(clip(B - lam, lo, up)) = 0.
%   h is nonincreasing in lam, so bisect. Scalar twin of the vectorized loop in
%   solve_mcsvm: same root, same answer.

    l = min(B) - Ci - 1;                          % h(l) >= 0
    h = max(B) + 1;                               % h(h) <= 0

    for k = 1:60
        mid = 0.5 * (l + h);
        if sum(min(up, max(lo, B - mid))) > 0
            l = mid;
        else
            h = mid;
        end
    end

    a = min(up, max(lo, B - 0.5 * (l + h)));
end


%% ======================================================================
%  Numerical helpers
%  ======================================================================

function [theta, tk] = momentum_step(opts, mu, L, tk, restartStat)
%MOMENTUM_STEP  FISTA momentum with adaptive restart.
%
%   restartStat = <z - x_new, x_new - x_old>. When positive, the momentum
%   direction is fighting the latest projected step, so restart.

    theta = 0;
    if ~opts.accel
        return;
    end

    if mu > 0
        % Strongly convex: constant momentum from the known modulus,
        % beta = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)). The gradient
        % restart is kept as a safeguard -- it never hurts, and it helps when
        % the local curvature on the active manifold exceeds the global mu.
        if restartStat > 0
            theta = 0;
        else
            rq    = sqrt(mu / L);
            theta = (1 - rq) / (1 + rq);
        end
    else
        % mu = 0: adaptive-restart FISTA.
        if restartStat > 0
            tk = 1;
        else
            tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
            theta = (tk - 1) / tk1;
            tk    = tk1;
        end
    end
end


function lam = bisect_root(h, scale)
%BISECT_ROOT  Find lam with h(lam) = 0, for h nonincreasing.
%
%   Brackets by doubling outward from [-scale, scale], then bisects.

    lo = -scale;
    hi =  scale;

    k = 0;
    while h(lo) < 0 && k < 60
        hi = lo;
        lo = 2 * lo;
        k  = k + 1;
    end

    k = 0;
    while h(hi) > 0 && k < 60
        lo = hi;
        hi = 2 * hi;
        k  = k + 1;
    end

    for k = 1:90
        mid = 0.5 * (lo + hi);
        if h(mid) > 0
            lo = mid;
        else
            hi = mid;
        end
    end

    lam = 0.5 * (lo + hi);
end


%% ======================================================================
%  Solver clock and histories
%
%  The clock is paused while the objective is recorded, so evaluation never
%  enters the reported time. It starts pre-charged with whatever setup that
%  particular solver actually needed -- see SETUP ACCOUNTING in the header.
%  ======================================================================

function clk = clk_new(charged)
    clk = struct('solve', charged, 'seg', tic);
end


function clk = clk_pause(clk)
    clk.solve = clk.solve + toc(clk.seg);
end


function clk = clk_resume(clk)
    clk.seg = tic;
end


function hist = init_hist()
    hist = struct('t', [], 'f', [], 'cols', []);
end


function hist = rec_hist(hist, t, f, cols)
    if nargin < 4
        cols = NaN;                      % un-instrumented recorder (LIBSVM)
    end
    hist.t(end+1, 1)    = t;
    hist.f(end+1, 1)    = f;
    hist.cols(end+1, 1) = cols;
end


function maybe_print(opts, name, it, f)
    if opts.verbose && mod(it - 1, opts.printEvery) == 0
        fprintf('%-16s step = %6d, obj = %.10e\n', name, it, f);
    end
end


%% ======================================================================
%  Figure
%  ======================================================================

function make_config_figure(hists, labels, figPath, ttl, ylab)
%MAKE_CONFIG_FIGURE  Log suboptimality against solver seconds.
%
%   CAVEAT: f* is taken as the best value any method in this figure reached.
%   That biases the plot toward whichever solver got furthest -- its own curve
%   is guaranteed to plunge to the floor while the others level off above zero.
%   For publication, compute f* once from an independent high-accuracy solve and
%   pass it in.

    fstar = inf;
    for i = 1:numel(hists)
        if ~isempty(hists{i}.f)
            fstar = min(fstar, min(hists{i}.f));
        end
    end
    floorVal = max(1e-16, 1e-12 * max(1, abs(fstar)));

    fig = figure('Visible', 'off');
    hold on;

    styles = {'r', 'b', 'k', 'g'};
    for i = 1:numel(hists)
        h = hists{i};
        if isempty(h.f)
            continue;
        end
        semilogy(h.t, max(h.f - fstar, floorVal), styles{min(i, numel(styles))}, ...
                 'LineWidth', 1.6, 'MarkerSize', 5, 'DisplayName', labels{i});
    end

    hold off;
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('Solver time (s)');
    ylabel(ylab);
    title(strrep(ttl, '_', '\_'));
    legend('Location', 'best');
    saveas(fig, figPath);
    legend('Location', 'best');
    saveas(fig, figPath);
    close(fig);
end

function results = gamma_sweep(matFile, opts0)
%GAMMA_SWEEP  Walk rho = sigma_1(K)/max_i K_ii down via rbfGamma, on the
%   UNMODIFIED box-only dual (biasMode='none'), and time PG vs DCD to a fixed
%   suboptimality target. Prints gamma | rho | PG-time | DCD-time | test-acc |
%   winner. rho is measured, not set; gamma is the only knob moved.
%
%   The squared-distance matrix is gamma-invariant and is built once here,
%   then reused for every gamma; see MAKE_KERNEL_OP for how its cost is
%   re-charged so the timings stay comparable to a standalone run.

    gammas = opts0.sweepGamma(:)';
    target = 1e-6;

    % Single source of truth for which variant the sweep runs.
    if isfield(opts0,'problem') && ~isempty(opts0.problem)
        prob = opts0.problem;
    else
        prob = 'mcsvm';            % sweep default; was hardcoded in the loop
    end
    isMC = strcmp(prob, 'mcsvm');

    sweep_log('gamma_sweep: %s\n', matFile);
    sweep_log('  problem = %s | gammas = [%s] | target = %.0e\n', ...
              prob, strtrim(sprintf('%g ', gammas)), target);

    tLoad = tic;
    [Xall, yraw] = load_xy_from_mat(matFile);
    if isMC
        [classes, ~, yall] = unique(yraw);   % yall in 1..K
        Kcls = numel(classes);
    else
        yall = sign(yraw); yall(yall==0) = 1;
        Kcls = 2;
    end
    sweep_log('  loaded n = %d, d = %d, K = %d in %.2f s\n', ...
              size(Xall,1), size(Xall,2), Kcls, toc(tLoad));

    rng(1); nAll = size(Xall,1); perm = randperm(nAll);
    nTr = round(0.8*nAll);
    trI = perm(1:nTr); teI = perm(nTr+1:end);

    % The split does not depend on gamma either, so hoist it with D2.
    Xtr = Xall(trI,:);  ytr = yall(trI);
    Xte = Xall(teI,:);  yte = yall(teI);

    % ---- gamma-invariant distance matrix, built ONCE ----------------------
    % Previously make_kernel_op recomputed full(X*X') and its n x n
    % temporaries for every gamma. On sparse text features that product is
    % the dominant setup cost and it is identical across the sweep.
    tD2 = tic;
    pre = struct('D2', sq_dists(Xtr, Xtr), 'd2Time', 0);
    pre.d2Time = toc(tD2);
    sweep_log('  squared distances %d x %d cached in %.2f s (%.1f GB resident)\n', ...
              size(pre.D2,1), size(pre.D2,2), pre.d2Time, 8*numel(pre.D2)/1e9);

    hdr1 = ' gamma |     rho |  PG (s) | DCD (s) | SMO (s) |  PG mv | DCD mv | acc%% | libAcc%% | winner\n';
    hdr2 = '-------+---------+---------+---------+---------+--------+--------+------+---------+--------\n';
    sweep_log(['\n' hdr1]);
    sweep_log(hdr2);

    rows     = [];
    rowLines = {};
    for g = gammas
        sweep_log('\n[gamma = %.2f]\n', g);

        o = opts0;
        o = rmfield(o, 'sweepGamma');       % avoid re-dispatch into this fn
        o.problem   = prob;
        o.biasMode  = 'none';               % box-only dual -> true problem
        o.rbfGamma  = g;
        o.makeFigure = false;
        o.overwrite  = true;
        o.verbose    = false;

        % Run the real pipeline on the TRAIN split only.
        r = run_one(Xtr, ytr, o, Kcls, pre);

        tPG  = time_to_target(r.pg.hist,  target);
        tDCD = time_to_target(r.dcd.hist, target);
        tSMO = time_to_target(r.smo.hist, target);

        % Machine-independent twin of the wall-clock crossover: gradient
        % equivalents to the same target (one full gradient = n, or nK).
        if isMC, denom = size(Xtr, 1) * Kcls; else, denom = size(Xtr, 1); end
        mvPG  = cols_to_target(r.pg.hist,  target) / denom;
        mvDCD = cols_to_target(r.dcd.hist, target) / denom;

        t0 = tic;
        if isMC
            acc = mc_test_acc(Xtr, ytr, Xte, yte, r.pg.alpha, g);
        else
            acc = rbf_test_acc(Xtr, ytr, Xte, yte, r.pg.alpha, g, o.C);
        end
        sweep_log('         acc  %8.2f s -> %.1f%%\n', toc(t0), 100*acc);

        t0 = tic;
        lacc = libsvm_test_acc(Xtr, ytr, Xte, yte, g, o.C);
        sweep_log('         ref  %8.2f s -> %.1f%% (LIBSVM, uncapped)\n', toc(t0), 100*lacc);

        % winner across all three
        cand = [tPG tDCD tSMO];
        names = {'PG','DCD','SMO'};
        if all(isnan(cand)), win = '--';
        else, [~,wi] = min(cand); win = names{wi};
        end

        rowStr = sprintf(' %5.2f | %7.2f | %7s | %7s | %7s | %6s | %6s | %4.1f | %5.1f | %s\n', ...
            g, r.rho, fmt(tPG), fmt(tDCD), fmt(tSMO), fmt(mvPG), fmt(mvDCD), ...
            100*acc, 100*lacc, win);
        sweep_log('%s', rowStr);

        rows     = [rows; g, r.rho, tPG, tDCD, mvPG, mvDCD, acc]; %#ok<AGROW>
        rowLines{end+1} = rowStr;                                 %#ok<AGROW>
    end

    % ---- clean reprint, now that the progress lines have broken up the table
    sweep_log('\n==== gamma_sweep summary ====\n');
    sweep_log(hdr1);
    sweep_log(hdr2);
    for i = 1:numel(rowLines)
        sweep_log('%s', rowLines{i});
    end

    results = struct('gammas', gammas, 'table', rows);
end

function r = run_one(X, y, opts, Kcls, pre)
%RUN_ONE  One gamma of the sweep: build the kernel, run PG / DCD / SMO.
%
%   PRE is the optional cached distance matrix; see MAKE_KERNEL_OP.
%
%   Each stage is timed and logged on completion. These wall-clock numbers
%   are diagnostic only -- the reported crossover still comes from
%   TIME_TO_TARGET on the recorded histories, which exclude objective
%   evaluation and include the setup charge.

    if nargin < 5
        pre = [];
    end

    opts = fill_default_opts(opts);
    isMC = strcmp(opts.problem, 'mcsvm');

    if isMC
        meta = struct('task','multiclass','K',Kcls);
    else
        meta = struct('task','binary','K',2);
    end
    P = make_problem(y, meta, opts);

    ker = make_kernel_op(X, y, P, opts, pre);

    P.setupGram = ker.gramTime;
    P.setupPG   = ker.gramTime + ker.sig1Time;

    % These two were missing: make_problem leaves them at 0, so every sweep
    % row reported gradient equivalents WITHOUT the setup charge that the
    % main run_svm_comparison path includes. The wall-clock and column axes
    % were not measuring the same thing.
    P.setupGramCols = ker.gramCols;
    P.setupPGCols   = ker.gramCols + ker.sig1Cols;

    r.rho   = ker.sig1 / 1.05;                    % RBF => max_i K_ii = 1
    r.tGram = ker.gramTime;
    r.tSig1 = ker.sig1Time;

    sweep_log('         Gram %8.2f s | sigma_1 %7.2f s | rho %8.2f\n', ...
              ker.gramTime, ker.sig1Time, r.rho);

    if isMC
        t0 = tic;  r.pg  = solve_mcsvm(ker, y, P, opts);              r.tPG  = toc(t0);
        sweep_log('         PG   %8.2f s (%d iters)\n', r.tPG, r.pg.iters);

        t0 = tic;  r.dcd = baseline_dcd_mcsvm(ker, X, y, P, opts);    r.tDCD = toc(t0);
        sweep_log('         DCD  %8.2f s\n', r.tDCD);

        t0 = tic;  r.smo = baseline_smo_mcsvm(ker, X, y, P, opts);    r.tSMO = toc(t0);
        sweep_log('         SMO  %8.2f s\n', r.tSMO);
    else
        t0 = tic;  r.pg  = solve_l1l2(ker, y, P, opts);               r.tPG  = toc(t0);
        sweep_log('         PG   %8.2f s (%d iters)\n', r.tPG, r.pg.iters);

        t0 = tic;  r.dcd = baseline_dcd_binary(ker, X, y, P, opts);   r.tDCD = toc(t0);
        sweep_log('         DCD  %8.2f s\n', r.tDCD);

        t0 = tic;  r.smo = baseline_libsvm_sweep(ker, X, y, P, opts); r.tSMO = toc(t0);
        sweep_log('         SMO  %8.2f s\n', r.tSMO);
    end
end

function sweep_log(varargin)
%SWEEP_LOG  Progress line for gamma_sweep / run_one, flushed as it is written.
%
%   gamma_sweep sets opts.verbose = false to silence the per-iteration solver
%   prints, which left the sweep silent for the full duration of a row. These
%   lines are the replacement: they are stage-level, not iteration-level, so
%   they stay quiet but make a long run distinguishable from a hung one.

    fprintf(varargin{:});
    if usejava('desktop')
        drawnow('limitrate');                     % force the desktop to flush
    end
end


function s = fmt(t); if isnan(t), s='  ---'; else, s=sprintf('%.2f',t); end; end

function t = time_to_target(hist, target)
%TIME_TO_TARGET  First recorded time at which f - f* <= target.
    fstar = min(hist.f);                    % panel-local; fine for crossover timing
    hit   = find(hist.f - fstar <= target, 1, 'first');
    if isempty(hit), t = NaN; else, t = hist.t(hit); end
end

function c = cols_to_target(hist, target)
%COLS_TO_TARGET  Kernel-column units at which f - f* first <= target.
%   NaN when the history is un-instrumented (LIBSVM) or never reaches target.
    fstar = min(hist.f);
    hit   = find(hist.f - fstar <= target, 1, 'first');
    if isempty(hit)
        c = NaN;
    else
        c = hist.cols(hit);
    end
end

function acc = rbf_test_acc(Xtr, ytr, Xte, yte, alpha, gamma, ~)
%RBF_TEST_ACC  Accuracy of the biasMode='none' L2-SVM classifier.
%   alpha is the multiplier on the SIGNED Gram, so the score already folds y in
%   as (y_tr .* alpha). No bias term exists under 'none' (b = 0).

    sv   = find(abs(alpha) > 0);
    if isempty(sv), acc = mean(yte == sign(0.5 - rand)); return; end  % degenerate

    coef = ytr(sv) .* alpha(sv);                 % signed coefficients
    Kte  = rbf_gram(Xte, Xtr(sv,:), gamma);      % nTe x |sv|, UNsigned kernel
    score = Kte * coef;                          % no bias under 'none'
    pred  = sign(score);  pred(pred == 0) = 1;
    acc   = mean(pred == yte);
end

function acc = mc_test_acc(Xtr, ytr, Xte, yte, alpha, gamma)
    sv = find(any(alpha ~= 0, 2));
    if isempty(sv), acc = mean(yte == mode(ytr)); return; end
    S = rbf_gram(Xte, Xtr(sv,:), gamma) * alpha(sv,:);   % nTe x K
    [~, pred] = max(S, [], 2);
    acc = mean(pred == yte);
end

function acc = libsvm_test_acc(Xtr, ytr, Xte, yte, gamma, C)
%LIBSVM_TEST_ACC  Reference accuracy from LIBSVM C-SVC with matched RBF.
%   Ground-truth cross-check for rbf_test_acc. NOTE: -s 0 is L1-SVM, not the
%   paper's L2 dual -- fine for a sign/ballpark check and for gamma selection,
%   which are insensitive to the L1/L2 distinction.
    if exist('svmtrain','file') ~= 3 && exist('./libsvm-336/matlab','dir')
        addpath('./libsvm-336/matlab');
    end
    if exist('svmtrain','file') ~= 3
        acc = NaN; return;                       % LIBSVM absent: skip, don't crash
    end
    model = svmtrain(ytr, sparse(Xtr), ...
                     sprintf('-s 0 -t 2 -g %g -c %g -q', gamma, C));
    pred  = svmpredict(yte, sparse(Xte), model, '-q');
    acc   = mean(pred == yte);
end


function results = cv_gamma_sweep(matFile, opts0)
%CV_GAMMA_SWEEP  k-fold CV accuracy vs gamma, to locate the generalization
%   peak relative to the PG/DCD crossover. Uses LIBSVM as the accuracy oracle
%   (matched RBF), so it is independent of our own test-accuracy convention and
%   doubles as the sign sanity-check. Prints gamma | rho | CV acc (mean+/-std).
    gammas = opts0.cvGamma(:)';
    k = 5; if isfield(opts0,'cvFolds') && ~isempty(opts0.cvFolds), k = opts0.cvFolds; end
    C = 1; if isfield(opts0,'C')       && ~isempty(opts0.C),       C = opts0.C;       end

    if exist('svmtrain','file') ~= 3 && exist('./libsvm-336/matlab','dir')
        addpath('./libsvm-336/matlab');
    end
    if exist('svmtrain','file') ~= 3
        error('cv_gamma_sweep: LIBSVM svmtrain mex not on path.');
    end

    [Xall, yall] = load_xy_from_mat(matFile);
    o0 = fill_default_opts(struct('problem','l2svm'));
    [Xall, yall] = preprocess_xy(Xall, yall, o0);   % same standardize + -/+1 as the sweep
    n = size(Xall,1);

    rng(1);
    fold = mod(randperm(n)' - 1, k) + 1;            % random fold labels 1..k

    fprintf('\n gamma |     rho | CV acc%% (mean +/- std)\n');
    fprintf('-------+---------+------------------------\n');

    accMean = zeros(size(gammas));
    for gi = 1:numel(gammas)
        g = gammas(gi);
        foldAcc = zeros(k,1);
        for f = 1:k
            teI = (fold == f);  trI = ~teI;
            model = svmtrain(yall(trI), sparse(Xall(trI,:)), ...
                             sprintf('-s 0 -t 2 -g %g -c %g -q', g, C));
            pred  = svmpredict(yall(teI), sparse(Xall(teI,:)), model, '-q');
            foldAcc(f) = mean(pred == yall(teI));
        end
        accMean(gi) = mean(foldAcc);
        rho = gamma_rho(Xall, g);
        fprintf(' %5.2f | %7.2f |     %5.1f +/- %4.1f\n', ...
                g, rho, 100*mean(foldAcc), 100*std(foldAcc));
    end

    [~, best] = max(accMean);
    fprintf('CV pick: gamma = %.2f  (%.1f%% held-out)\n', gammas(best), 100*accMean(best));
    results = struct('gammas', gammas, 'cvAcc', accMean, 'pick', gammas(best));
end


function rho = gamma_rho(X, gamma)
%GAMMA_RHO  Measured sigma_1(K)/max_i K_ii for one gamma (no safety factor).
    K = rbf_gram(X, X, gamma);
    optsE.tol = 1e-3; optsE.issym = true; optsE.isreal = true;
    s1  = eigs(@(a) K*a, size(K,1), 1, 'largestabs', optsE);
    rho = abs(s1) / max(full(diag(K)));
end
