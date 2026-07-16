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
%       opts.kernel      'rbf' (default) | 'linear'
%       opts.costMode    'none' | 'cost'   (per-class C_i weighting)
%       opts.biasMode    'constrained' | 'none' | 'augmented'
%       opts.timeLimit   solver wall-clock budget, seconds
%
%   Every solver is timed on the same clock, which is paused while the
%   objective is recorded, and which starts pre-charged with the setup cost
%   that solver actually incurs (see SETUP ACCOUNTING below).
%
%   BIAS AND THE DUAL EQUALITY CONSTRAINT
%       The l1/l2/svr duals carry an equality constraint (<alpha,y> = 0 or
%       1'beta = 0) that comes from the primal bias term b. Single-coordinate
%       descent cannot move on such a dual, so opts.biasMode selects the
%       formulation:
%
%         'constrained'  keep the equality. PG bisects for the multiplier;
%                        the coordinate baselines skip themselves.
%         'none'         no bias at all (LIBLINEAR -B -1, and the setting used
%                        by Hsieh et al. 2008). Box-only dual, both families
%                        apply, K is untouched.
%         'augmented'    penalized bias via a rank-one Gram shift
%                        Ktilde = K + s^2 v v', s = opts.biasScale. Box-only
%                        dual. Note sigma_1(Ktilde) ~ sigma_1(K) + s^2 n, so s
%                        trades model fidelity against conditioning; s = 0 is
%                        exactly 'none'.
%
%       nusvm is exempt: its two class-mass equalities define nu and cannot be
%       reformulated away, so its coordinate baseline uses same-class PAIRS.
%       mcsvm is exempt: Crammer-Singer has no bias term, and its equality is
%       per-row, so block coordinate descent applies as-is.
%
%   SETUP ACCOUNTING
%       ker.gramTime  building K.       Needed by PG and by every coordinate
%                                       baseline that reads ker.K. Shared.
%       ker.sig1Time  power iteration.  Needed by PG only -- no coordinate
%                                       method reads ker.sig1.
%
%       PG solvers are charged P.setupPG = gramTime + sig1Time.
%       Coordinate and SMO baselines that read ker.K are charged
%       P.setupGram = gramTime.
%       LIBSVM is charged only for kernels it cannot build itself: with -t 0 or
%       -t 2 it caches internally inside its own timed region (nothing charged);
%       with -t 4 (l2svm) it needs a precomputed Gram, which is timed and added
%       to every recorded point.
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
    addpath('./liblinear-249/matlab');

    if nargin < 2
        opts = struct();
    end
    opts = fill_default_opts(opts);
    rng(opts.seed);

    % ---- output path -----------------------------------------------------
    [~, datasetName] = fileparts(matFile);
    figDir  = fullfile(opts.outRoot, datasetName);
    figPath = fullfile(figDir, sprintf('%s_%s.png', opts.problem, opts.costMode));

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

    fprintf('%s | n = %d, d = %d | problem = %s | costMode = %s | C = %g\n', ...
            datasetName, size(X, 1), size(X, 2), opts.problem, opts.costMode, opts.C);

    % ---- kernel operator -------------------------------------------------
    ker = make_kernel_op(X, y, P, opts);
    report_conditioning(ker, X, opts);

    ker = augment_kernel_bias(ker, y, P, opts);

    P.setupGram = ker.gramTime;                   % billed to every ker.K reader
    P.setupPG   = ker.gramTime + ker.sig1Time;    % PG also pays for sigma_1
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
    switch P.name
        case 'mcsvm'
            cand = { baseline_smo_mcsvm(ker, X, y, P, opts),    'CS SMO (max-violating pair)'
                     baseline_dcd_mcsvm(ker, X, y, P, opts),    'CS sequential dual (Keerthi et al. 2008)' };
        case 'nusvm'
            cand = { baseline_libsvm_sweep(ker, X, y, P, opts), 'LIBSVM SMO'
                     baseline_pcd_nusvm(ker, X, y, P, opts),    'Pairwise dual CD (same-class SMO)' };
        case 'svr'
            cand = { baseline_libsvm_sweep(ker, X, y, P, opts), 'LIBSVM SMO'
                     baseline_dcd_svr(ker, X, y, P, opts),      'Kernel dual CD (SVR)' };
        case {'l1svm', 'l2svm'}
            cand = { baseline_libsvm_sweep(ker, X, y, P, opts), 'LIBSVM SMO'
                     baseline_dcd_binary(ker, X, y, P, opts),   'Kernel dual CD / SOR' };
        otherwise
            cand = { baseline_libsvm_sweep(ker, X, y, P, opts), 'LIBSVM SMO' };
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
        ttl = sprintf('%s: %s (%s)', datasetName, P.name, opts.costMode);
        make_config_figure(hists, labels, figPath, ttl, 'Dual suboptimality  f - f*');
        fprintf('Saved %s\n', figPath);
    end

    results          = struct();
    results.skipped  = false;
    results.figPath  = figPath;
    results.problem  = P.name;
    results.costMode = opts.costMode;
    results.proposed = propOut;
    results.baseline = baseOuts;     % cell array, aligned with labels(2:end)
    results.labels   = labels;
    results.hists    = hists;
    results.opts     = opts;
end


function report_conditioning(ker, X, opts)
%REPORT_CONDITIONING  Print sigma_1(K) / max_i K_ii.
%
%   The projected-gradient step is 1/sigma_1(K); the exact coordinate step is
%   1/K_ii. This ratio is therefore the per-coordinate handicap the global
%   Lipschitz constant imposes, and it predicts which family wins. Report it
%   before any bias shift is applied, so it describes the true Gram.
%
%   Note ker.sig1 carries a 1.05 safety factor from the power iteration; it is
%   divided out here so the printed value is the actual spectral radius.

    if ker.explicit
        maxKii = max(full(diag(ker.K)));
    elseif strcmp(opts.kernel, 'linear')
        maxKii = max(full(sum(X .^ 2, 2)));
    else
        maxKii = 1;                              % RBF: k(x, x) = exp(0) = 1
    end

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
    opts = set_default(opts, 'problem',  'nusvm');   % l1svm l2svm svr nusvm mcsvm
    opts = set_default(opts, 'costMode', 'none');
    opts = set_default(opts, 'C',        1);
    opts = set_default(opts, 'nu',       0.1);
    opts = set_default(opts, 'epsSVR',   0.1);

    % kernel
    opts = set_default(opts, 'kernel',   'rbf');     % 'linear' | 'rbf'
    opts = set_default(opts, 'rbfGamma', 0.5);       % k = exp(-g||x - x'||^2),
                                                     % g = 1/(2r^2), radius 1
    opts = set_default(opts, 'explicitKernelMaxN', 80000);

    % bias handling (see the header of run_svm_comparison)
    opts = set_default(opts, 'biasMode',  'none');   % 'constrained' | 'none' | 'augmented'
    opts = set_default(opts, 'biasScale', 1);        % s in Ktilde = K + s^2 v v'

    % proximal solver
    opts = set_default(opts, 'accel',       true);
    opts = set_default(opts, 'lazy',        true);   % incremental K*delta updates
    opts = set_default(opts, 'lazyRefresh', 10);    % full recompute cadence (FP drift)
    opts = set_default(opts, 'maxIters',    5000);
    opts = set_default(opts, 'tol',         1e-12);
    opts = set_default(opts, 'timeLimit',   60);

    % cost-sensitive weighting
    opts = set_default(opts, 'classCosts',    []);   % per-class multipliers
    opts = set_default(opts, 'costPosFactor', 2);    % svr 'cost': over-estimate box
    opts = set_default(opts, 'costNegFactor', 1);    % svr 'cost': under-estimate box

    % task adaptation
    opts = set_default(opts, 'task',         '');
    opts = set_default(opts, 'binarize',     struct('type', 'halfsplit'));
    opts = set_default(opts, 'mcBins',       4);
    opts = set_default(opts, 'maxSamples',   inf);
    opts = set_default(opts, 'standardize',  true);
    opts = set_default(opts, 'standardizeY', true);

    % baselines
    opts = set_default(opts, 'smoTolerances', 10 .^ -(1:8));
    opts = set_default(opts, 'liblinearFn',   'train');

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

    valid = {'constrained', 'augmented', 'none'};
    if ~any(strcmp(opts.biasMode, valid))
        error('opts.biasMode must be ''constrained'', ''augmented'', or ''none''.');
    end

    if strcmp(opts.biasMode, 'none')
        opts.biasScale = 0;      % zero shift == bias-free dual (LIBLINEAR -B -1)
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
    % destroy sparsity and blow up memory.
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
    P.setupGram = 0;
    P.setupPG   = 0;

    withCost = strcmp(opts.costMode, 'cost');

    % P.hasEq is read only by solve_l1l2, solve_svr, baseline_dcd_binary and
    % baseline_dcd_svr. It means "the dual carries a bias equality that the
    % projection must enforce" -- not "the dual has any equality at all".
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

function ker = make_kernel_op(X, y, P, opts)
%MAKE_KERNEL_OP  Build K*a, sigma_1(K), and (when affordable) K itself.
%
%   Returns
%       ker.mul       @(a) -> K*a. Accepts n x 1 and n x K right-hand sides.
%       ker.explicit  is ker.K materialized?
%       ker.K         the n x n Gram (explicit mode only)
%       ker.sig1      1.05 * lambda_max(K), capped at n for RBF
%       ker.gramTime  seconds spent building K       (shared cost)
%       ker.sig1Time  seconds spent on sigma_1       (PG-only cost)
%
%   For l1svm/l2svm/nusvm the Gram is SIGNED: K = (y y') .* Kraw. Callers that
%   index ker.K must account for that.
%
%   Every coordinate baseline needs kernel COLUMNS, so they require
%   ker.explicit. In practice that caps n in the low thousands for RBF, since a
%   dense n x n double is 8n^2 bytes.

    n      = size(X, 1);
    signed = any(strcmp(P.name, {'l1svm', 'l2svm', 'nusvm'}));
    isRbf  = strcmp(opts.kernel, 'rbf');

    % ---- Gram ------------------------------------------------------------
    tGram = tic;

    if n <= opts.explicitKernelMaxN
        if isRbf
            K = rbf_gram(X, X, opts.rbfGamma);
        else
            K = full(X * X');
        end
        if signed
            K = (y * y') .* K;
        end
        ker.mul      = @(a) K * a;
        ker.explicit = true;
        ker.K        = K;

    elseif isRbf
        warning('n = %d > explicitKernelMaxN: RBF K*a computed in blocks (slow).', n);
        if signed
            ker.mul = @(a) y .* rbf_mul_blocked(X, opts.rbfGamma, y .* a);
        else
            ker.mul = @(a) rbf_mul_blocked(X, opts.rbfGamma, a);
        end
        ker.explicit = false;

    else
        Xt = X';                                  % cache the transpose once
        if signed
            ker.mul = @(a) y .* (X * (Xt * (y .* a)));
        else
            ker.mul = @(a) X * (Xt * a);
        end
        ker.explicit = false;
    end

    ker.gramTime = toc(tGram);

    % ---- sigma_1 ---------------------------------------------------------
    tSig1 = tic;

    % Every PG solver uses adaptive backtracking, so we just need a
    % reasonable lower bound to start the local Lipschitz estimate.
    if ker.explicit
        ker.sig1 = max(full(diag(ker.K)));
    else
        ker.sig1 = 1;
    end

    ker.sig1Time  = toc(tSig1);
    ker.setupTime = ker.gramTime + ker.sig1Time;
end


function K = rbf_gram(A, B, gamma)
%RBF_GRAM  k(a, b) = exp(-gamma ||a - b||^2).
%
%   gamma = 1/(2 radius^2) matches LIBSVM's -g convention, so radius 1 is
%   gamma 0.5.

    sqA = full(sum(A .^ 2, 2));
    sqB = full(sum(B .^ 2, 2));
    D2  = bsxfun(@plus, sqA, sqB') - 2 * full(A * B');
    K   = exp(-gamma * max(D2, 0));               % clamp negative round-off
end


function out = rbf_mul_blocked(X, gamma, a)
%RBF_MUL_BLOCKED  K*a for the RBF kernel without materializing K.
%
%   Rows are processed in blocks of roughly 1 GB of doubles. Accepts n x 1 and
%   n x K right-hand sides.
%
%   Sparse `a` -- which is what the lazy delta updates hand us -- takes a fast
%   path: the Gram is formed only against the support of a, so the cost is
%   O(n |supp(a)| d) rather than O(n^2 d).

    n = size(X, 1);

    if issparse(a)
        idx = find(any(a ~= 0, 2));
        if isempty(idx)
            out = zeros(n, size(a, 2));
            return;
        end

        Xs  = X(idx, :);
        af  = full(a(idx, :));
        blk = max(256, floor(1.25e8 / max(1, numel(idx))));
        out = zeros(n, size(a, 2));
        for i0 = 1:blk:n
            ii = i0:min(i0 + blk - 1, n);
            out(ii, :) = rbf_gram(X(ii, :), Xs, gamma) * af;
        end
        return;
    end

    blk = max(256, floor(1.25e8 / n));
    out = zeros(n, size(a, 2));
    for i0 = 1:blk:n
        ii = i0:min(i0 + blk - 1, n);
        out(ii, :) = rbf_gram(X(ii, :), X, gamma) * a;
    end
end


function ker = augment_kernel_bias(ker, y, P, opts)
%AUGMENT_KERNEL_BIAS  Rank-one Gram shift for the penalized-bias dual.
%
%   Appending a constant feature s to phi(x) is impossible for an RBF map, but
%   its effect on the Gram is not: it is the rank-one shift
%
%       Ktilde = K + s^2 v v',    Ktilde_ii = K_ii + s^2
%
%   with v = y for the signed problems and v = 1 for svr. The bias is then
%   regularized as (1/2)(b/s)^2, the dual equality vanishes, and the feasible
%   set becomes a pure box -- which is exactly what single-coordinate descent
%   needs.
%
%   The shift is rank one and is never materialized: it is folded into ker.mul
%   (and into kacc.col / kacc.diag by dcd_kernel_ops) at O(n) extra cost per
%   matvec. sigma_1 is recomputed for the shifted operator, because solve_l1l2
%   reads ker.sig1 for its step size and a stale value would let it overstep.
%
%   Scale matters. lambda_max(s^2 v v') = s^2 ||v||^2 = s^2 n, so
%   sigma_1(Ktilde) ~ sigma_1(K) + s^2 n. On a well-conditioned Gram, s = 1 can
%   make the shift the entire Lipschitz constant -- crippling the PG step while
%   barely touching the coordinate step (K_ii goes from 1 to 1 + s^2). Sweeping
%   s is therefore a clean way to walk the conditioning ratio from sigma_1(K)
%   up to n on fixed data. s = 0 is exactly biasMode 'none'.

    ker.b2 = 0;
    ker.bv = [];

    % mcsvm has no bias term. nusvm has two coupled class-mass equalities that a
    % bias shift does not remove, so shifting K while solve_nusvm still projects
    % onto them would silently solve a different problem.
    if any(strcmp(P.name, {'mcsvm', 'nusvm'}))
        return;
    end
    if ~any(strcmp(opts.biasMode, {'augmented', 'none'}))
        return;                                   % 'constrained': keep the equality
    end

    s2 = opts.biasScale ^ 2;
    if s2 == 0
        fprintf('biasMode = none: bias-free dual, no shift, sigma_1(K) = %.4e\n', ker.sig1);
        return;                                   % ker.sig1 is already correct
    end

    n = numel(y);
    if any(strcmp(P.name, {'l1svm', 'l2svm'}))
        v = y(:);                                 % signed Gram
    else
        v = ones(n, 1);                           % unsigned (svr)
    end

    baseMul = ker.mul;
    ker.mul = @(a) baseMul(a) + s2 * (v * (v' * a));   % rank one; n x K safe
    ker.b2  = s2;
    ker.bv  = v;

    tSig1 = tic;
    u = randn(n, 1);
    u = u / norm(u);
    for k = 1:100
        u = ker.mul(u);
        u = u / max(norm(u), 1e-300);
    end
    ker.sig1      = max(1.05 * (u' * ker.mul(u)), 1e-12);
    ker.sig1Time  = ker.sig1Time + toc(tSig1);
    ker.setupTime = ker.gramTime + ker.sig1Time;

    fprintf(['biasMode = augmented: rank-one shift s^2 = %.4g applied; ', ...
             'equality dropped, sigma_1(Ktilde) = %.4e\n'], s2, ker.sig1);
end


function kacc = dcd_kernel_ops(ker, X, y, P, opts)   %#ok<INUSL>
%DCD_KERNEL_OPS  Column / diagonal access for the coordinate methods.
%
%   kacc.mode   'explicit'  Gram cached. A coordinate gradient is one column
%                           lookup, O(n). This is the mode to use.
%               'linear'    No Gram, but a linear kernel, so maintain
%                           w = X'(y .* a) and read the gradient in O(nnz(x_i)).
%               'blocked'   RBF, no Gram. Every coordinate would need a fresh
%                           kernel column, O(nd), making an epoch O(n^2 d).
%                           Callers skip themselves in this mode rather than
%                           report a meaningless wall-clock number.
%
%   kacc.diag   n x 1, diag(Ktilde)
%   kacc.col    @(i) -> Ktilde(:, i)          (explicit mode only)
%   kacc.b2     rank-one bias shift magnitude (0 if none)
%   kacc.bv     rank-one bias shift vector    ([] if none)
%
%   Note kacc.mode is derived from ker.explicit AND opts.kernel; it is not a
%   restatement of opts.kernel. A linear kernel on small n gives 'explicit',
%   because a cached column beats the w-trick.

    n  = numel(y);
    b2 = 0;
    bv = [];
    if isfield(ker, 'b2') && ~isempty(ker.b2)
        b2 = ker.b2;
        bv = ker.bv;
    end

    kacc = struct('b2', b2, 'bv', bv);

    if ker.explicit
        kacc.mode = 'explicit';
        kacc.diag = full(diag(ker.K));
        if b2 > 0
            kacc.diag = kacc.diag + b2 * (bv .^ 2);
            kacc.col  = @(i) ker.K(:, i) + (b2 * bv(i)) * bv;
        else
            kacc.col  = @(i) ker.K(:, i);
        end

    elseif strcmp(opts.kernel, 'linear')
        kacc.mode = 'linear';
        kacc.diag = full(sum(X .^ 2, 2));
        if b2 > 0
            kacc.diag = kacc.diag + b2 * (bv .^ 2);
        end
        kacc.col = [];

    else
        kacc.mode = 'blocked';
        kacc.diag = ones(n, 1);                   % RBF: k(x, x) = 1
        kacc.col  = [];
    end

    kacc.diag = max(kacc.diag, 1e-12);
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
%   Under biasMode 'none'/'augmented' the equality is gone, lambda = 0, and the
%   projection collapses to a clip.

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
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end


        % ---- gradient step, then project ---------------------------------
        eta    = 2;      % backtracking growth
        shrink = 0.98;   % let L drift back down on the active manifold

        fZ    = plotted_objective(P, z, gz, y);
        if isL2
            gradZ = gz + z ./ P.Ci - 1;
        else
            gradZ = gz - 1;
        end

        while true
            % 1. Gradient descent step
            beta = z - gradZ / L;

            % 2. Project beta to get alphaNew
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

            % 3. Lazy update & evaluate
            dA    = alphaNew - alpha;
            KdA   = ker.mul(sparse(dA));
            GAnew = ga + KdA;

            fNew  = plotted_objective(P, alphaNew, GAnew, y);

            D = alphaNew - z;
            Q = fZ + sum(D .* gradZ) + (L / 2) * sum(D .^ 2);

            if fNew <= Q + 1e-12 * max(1, abs(fZ))
                break;
            end
            L = eta * L;
        end
        L = max(L * shrink, 1e-12);
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
        sinceRefresh = sinceRefresh + 1;
        gaPrev       = ga;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * n
            ga = GAnew;
        else
            ga           = ker.mul(alphaNew);    % periodic full refresh
            sinceRefresh = 0;
        end
        gz = ga + theta * (ga - gaPrev);

        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));

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

    n = numel(y);

    if ker.explicit
        L = max(full(diag(ker.K)));
    else
        L = 1;  % RBF k(x,x)=1, or a safe generic lower bound
    end
    eta    = 2;     % backtracking growth
    shrink = 0.98;  % let L drift back down on the active manifold

    b            = zeros(n, 1);
    z            = b;
    tk           = 1;
    gb           = zeros(n, 1);          % maintained K*b
    gz           = zeros(n, 1);          % maintained K*z
    sinceRefresh = 0;

    hist = init_hist();
    clk  = clk_new(P.setupPG);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f    = plotted_objective(P, b, gb, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end

        % --- Objective and Gradient at z ---
        fz    = plotted_objective(P, z, gz, y);
        gradZ = gz - y;

        % --- Backtracking Line Search ---
        while true
            % The soft-threshold depends on L, so it must be updated
            % inside the loop whenever L changes
            thr = P.eps / L;
            st  = @(u) sign(u) .* max(abs(u) - thr, 0);   % S_{eps/L}

            % 1. Gradient descent step
            v = z - gradZ / L;

            % 2. Project v to get bNew
            if P.hasEq
                h   = @(lam) sum(min(P.bUp, max(P.bLo, st(v - lam))));
                lam = bisect_root(h, max(1, max(abs(v))));
            else
                lam = 0;
            end
            bNew = min(P.bUp, max(P.bLo, st(v - lam)));

            % 3. Lazy update & evaluate
            dB    = bNew - b;
            KdB   = ker.mul(sparse(dB));
            gbNew = gb + KdB;

            fNew  = plotted_objective(P, bNew, gbNew, y);

            D = bNew - z;
            Q = fz + sum(D .* gradZ) + (L / 2) * sum(D .^ 2);

            if fNew <= Q + 1e-12 * max(1, abs(fz))
                break; % accept
            end
            L = eta * L; % half step
        end
        L = max(L * shrink, 1e-12);

        converged = norm(dB) <= opts.tol * max(1, norm(b));

        % --- Momentum ---
        theta = 0;
        if opts.accel
            if (z - bNew)' * dB > 0 % gradient restart
                tk = 1;
            else
                tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
                theta = (tk - 1) / tk1;
                tk    = tk1;
            end
        end
        z = bNew + theta * dB;

        % --- Lazy kernel-image maintenance ---
        sinceRefresh = sinceRefresh + 1;
        gbPrev       = gb;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dB) < 0.5 * n
            gb = gbNew; % reuse from backtrace
        else
            gb           = ker.mul(bNew);
            sinceRefresh = 0;
        end
        gz = gb + theta * (gb - gbPrev);

        b = bNew;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, b, ker.mul(b), y));

    out = struct('alpha', b, 'hist', hist, 'iters', it, 'skipped', false);
end
% function out = solve_svr(ker, y, P, opts)
% %SOLVE_SVR  Accelerated proximal gradient for the eps-insensitive SVR dual.
% %
% %   Eq. 10:  min_b  0.5 b'Kb - y'b + eps ||b||_1
% %            s.t.   bLo <= b <= bUp,  1'b = 0
% %
% %   Iteration, with L = sigma_1(K) and S the soft-threshold at eps/L:
% %
% %       v      <- z - (K z - y)/L
% %       lambda <- root of  sum_i clip(S(v_i - lambda)) = 0     (bisection)
% %       b      <- clip(S(v - lambda))

%     n   = numel(y);
%     L   = ker.sig1;
%     thr = P.eps / L;
%     st  = @(u) sign(u) .* max(abs(u) - thr, 0);   % S_{eps/L}

%     b            = zeros(n, 1);
%     z            = b;
%     tk           = 1;
%     gb           = zeros(n, 1);          % maintained K*b
%     gz           = zeros(n, 1);          % maintained K*z
%     sinceRefresh = 0;

%     hist = init_hist();
%     clk  = clk_new(P.setupPG);

%     it = 0;
%     while it < opts.maxIters
%         it = it + 1;

%         clk = clk_pause(clk);
%         if mod(it - 1, opts.evalEvery) == 0
%             f    = plotted_objective(P, b, gb, y);
%             hist = rec_hist(hist, clk.solve, f);
%             maybe_print(opts, P.name, it, f);
%         end
%         stop = clk.solve >= opts.timeLimit;
%         clk  = clk_resume(clk);
%         if stop
%             break;
%         end

%         v = z - (gz - y) / L;

%         if P.hasEq
%             h   = @(lam) sum(min(P.bUp, max(P.bLo, st(v - lam))));
%             lam = bisect_root(h, max(1, max(abs(v))));
%         else
%             lam = 0;
%         end
%         bNew = min(P.bUp, max(P.bLo, st(v - lam)));

%         dB        = bNew - b;
%         converged = norm(dB) <= opts.tol * max(1, norm(b));

%         theta = 0;
%         if opts.accel
%             if (z - bNew)' * dB > 0              % gradient restart
%                 tk = 1;
%             else
%                 tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
%                 theta = (tk - 1) / tk1;
%                 tk    = tk1;
%             end
%         end
%         z = bNew + theta * dB;

%         sinceRefresh = sinceRefresh + 1;
%         gbPrev       = gb;

%         if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dB) < 0.5 * n
%             gb = gb + ker.mul(sparse(dB));
%         else
%             gb           = ker.mul(bNew);
%             sinceRefresh = 0;
%         end
%         gz = gb + theta * (gb - gbPrev);

%         b = bNew;
%         if converged
%             break;
%         end
%     end

%     clk  = clk_pause(clk);
%     hist = rec_hist(hist, clk.solve, plotted_objective(P, b, ker.mul(b), y));

%     out = struct('alpha', b, 'hist', hist, 'iters', it, 'skipped', false);
% end

function out = solve_mcsvm(ker, y, P, opts)
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

    % --- Initializing the local Lipschitz estimate ---
    if ker.explicit
        L = max(full(diag(ker.K)));
    else
        L = 1;  % RBF k(x,x)=1, or a safe generic lower bound
    end
    eta    = 2;     % backtracking growth
    shrink = 0.98;  % let L drift back down on the active manifold

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

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f    = plotted_objective(P, alpha, GA, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end

        % --- Objective and Gradient at Z ---
        fZ    = 0.5 * sum(sum(Z .* GZ)) - sum(sum(Z .* E));
        gradZ = GZ - E;

        % --- Backtracking Line Search ---
        while true
            B = Z - gradZ / L;

            % ---- Eq. 16 for every row at once --------------------------------
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

            dA = alphaNew - alpha;

            % 3. Lazy update & evaluate
            KdA   = ker.mul(sparse(dA));
            GAnew = GA + KdA;
            fNew  = 0.5 * sum(sum(alphaNew .* GAnew)) - sum(sum(alphaNew .* E));

            D = alphaNew - Z;
            Q = fZ + sum(sum(D .* gradZ)) + (L / 2) * sum(sum(D .^ 2));

            if fNew <= Q + 1e-12 * max(1, abs(fZ))
                break;                                   % accepted
            end
            L = eta * L;                                 % too bold; halve the step
        end
        L = max(L * shrink, 1e-12);                      % allow downward adaptation

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

        sinceRefresh = sinceRefresh + 1;
        GAPrev       = GA;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * numel(dA)
            GA = GAnew;
        else
            GA           = ker.mul(alphaNew);
            sinceRefresh = 0;
        end
        GZ = GA + theta * (GA - GAPrev);

        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));

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
    up = P.up;
    ip = (y > 0);
    im = ~ip;

    % --- Initializing the local Lipschitz estimate ---
    if ker.explicit
        L = max(full(diag(ker.K)));
    else
        L = 1;  % RBF k(x,x)=1, or a safe generic lower bound
    end
    eta    = 2;     % backtracking growth
    shrink = 0.98;  % let L drift back down on the active manifold

    alpha        = zeros(n, 1);
    z            = alpha;
    tk           = 1;
    ga           = zeros(n, 1);          % maintained K*alpha
    gz           = zeros(n, 1);          % maintained K*z
    sinceRefresh = 0;

    hist = init_hist();
    clk  = clk_new(P.setupPG);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        if it > 1 && mod(it - 1, opts.evalEvery) == 0
            f    = plotted_objective(P, alpha, ga, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk  = clk_resume(clk);
        if stop
            break;
        end

        % --- Objective and Gradient at z ---
        fz    = plotted_objective(P, z, gz, y);
        gradZ = gz;

        % --- Backtracking Line Search ---
        while true
            B = z - gradZ / L;

            % ---- Eq. 21: project onto {0 <= a <= up, y'a = 0} -----------------
            h   = @(lam) sum(y .* min(up, max(B - lam * y, 0)));
            lam = bisect_root(h, max(1, max(abs(B))));
            aNew = min(up, max(B - lam * y, 0));

            % ---- Eqs. 22-23: enforce nu/2 of mass in each class ---------------
            if sum(aNew) < P.nu - 1e-12
                hp = @(l) sum(min(up(ip), max(B(ip) - l, 0))) - P.nu / 2;
                hm = @(l) sum(min(up(im), max(B(im) - l, 0))) - P.nu / 2;
                lp = bisect_root(hp, max(1, max(abs(B(ip)))));
                lm = bisect_root(hm, max(1, max(abs(B(im)))));

                aNew(ip) = min(up(ip), max(B(ip) - lp, 0));
                aNew(im) = min(up(im), max(B(im) - lm, 0));
            end

            % 3. Lazy update & evaluate
            dA    = aNew - alpha;
            KdA   = ker.mul(sparse(dA));
            gaNew = ga + KdA;

            fNew  = plotted_objective(P, aNew, gaNew, y);

            D = aNew - z;
            Q = fz + sum(D .* gradZ) + (L / 2) * sum(D .^ 2);

            if fNew <= Q + 1e-12 * max(1, abs(fz))
                break;
            end
            L = eta * L;
        end
        L = max(L * shrink, 1e-12);

        converged = norm(dA) <= opts.tol * max(1, norm(alpha));

        theta = 0;
        if opts.accel
            if (z - aNew)' * dA > 0                  % gradient restart
                tk = 1;
            else
                tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
                theta = (tk - 1) / tk1;
                tk    = tk1;
            end
        end
        z = aNew + theta * dA;

        sinceRefresh = sinceRefresh + 1;
        gaPrev       = ga;

        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * n
            ga = gaNew;
        else
            ga           = ker.mul(aNew);
            sinceRefresh = 0;
        end
        gz = ga + theta * (ga - gaPrev);

        alpha = aNew;
        if converged
            break;
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end

% function out = solve_mcsvm(ker, y, P, opts)
% %SOLVE_MCSVM  Jacobi-style parallel projected gradient for Crammer-Singer.
% %
% %   min_a  0.5 tr(a'Ka) - <a, E>
% %   s.t.   0 <= a_{i,y_i} <= C_i,  -C_i <= a_{i,j} <= 0 (j != y_i),  a_i 1 = 0
% %
% %   Iteration, with L = sigma_1(K):
% %
% %       B <- Z - (K Z - E)/L
% %       for each row i (in parallel): find lambda_i satisfying Eq. 16 by
% %       bisection, then a_i <- clip(B_i - lambda_i)
% %
% %   The row equality is per-row, so all n root-finds are independent and the
% %   bisection below runs them simultaneously.

%     n = numel(y);
%     K = P.K;
%     L = ker.sig1;

%     E   = full(sparse((1:n)', y, 1, n, K));       % 1_ind
%     CiK = repmat(P.Ci, 1, K);
%     UP  = E .* CiK;                               % [0, C_i] at the true class
%     LO  = (E - 1) .* CiK;                         % [-C_i, 0] elsewhere

%     alpha        = zeros(n, K);
%     Z            = alpha;
%     tk           = 1;
%     GA           = zeros(n, K);          % maintained K*alpha
%     GZ           = zeros(n, K);          % maintained K*Z
%     sinceRefresh = 0;

%     hist = init_hist();
%     clk  = clk_new(P.setupPG);

%     it = 0;
%     while it < opts.maxIters
%         it = it + 1;

%         clk = clk_pause(clk);
%         if mod(it - 1, opts.evalEvery) == 0
%             f    = plotted_objective(P, alpha, GA, y);   % free: GA is maintained
%             hist = rec_hist(hist, clk.solve, f);
%             maybe_print(opts, P.name, it, f);
%         end
%         stop = clk.solve >= opts.timeLimit;
%         clk  = clk_resume(clk);
%         if stop
%             break;
%         end

%         B = Z - (GZ - E) / L;

%         % ---- Eq. 16 for every row at once --------------------------------
%         % Each h_i is nonincreasing in lambda_i, positive at lo and nonpositive
%         % at hi, so one vectorized bisection resolves all n roots together.
%         lo = min(B, [], 2) - max(P.Ci) - 1;
%         hi = max(B, [], 2) + 1;
%         for k = 1:80
%             mid      = (lo + hi) / 2;
%             Cl       = min(UP, max(LO, bsxfun(@minus, B, mid)));
%             pos      = sum(Cl, 2) > 0;
%             lo(pos)  = mid(pos);
%             hi(~pos) = mid(~pos);
%         end
%         mid      = (lo + hi) / 2;
%         alphaNew = min(UP, max(LO, bsxfun(@minus, B, mid)));

%         dA        = alphaNew - alpha;
%         converged = norm(dA(:)) <= opts.tol * max(1, norm(alpha(:)));

%         theta = 0;
%         if opts.accel
%             if sum(sum((Z - alphaNew) .* dA)) > 0    % gradient restart
%                 tk = 1;
%             else
%                 tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
%                 theta = (tk - 1) / tk1;
%                 tk    = tk1;
%             end
%         end
%         Z = alphaNew + theta * dA;

%         % Rows whose every coordinate stayed clipped at its bound have an
%         % all-zero delta row, so K*dA touches only the active rows.
%         sinceRefresh = sinceRefresh + 1;
%         GAPrev       = GA;

%         if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * numel(dA)
%             GA = GA + ker.mul(sparse(dA));
%         else
%             GA           = ker.mul(alphaNew);
%             sinceRefresh = 0;
%         end
%         GZ = GA + theta * (GA - GAPrev);

%         alpha = alphaNew;
%         if converged
%             break;
%         end
%     end

%     clk  = clk_pause(clk);
%     hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));

%     out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
% end


% function out = solve_nusvm(ker, y, P, opts)
% %SOLVE_NUSVM  Accelerated projected gradient for the nu-SVM dual.
% %
% %   min_a  0.5 a'Ka
% %   s.t.   0 <= a <= up,  <a, y> = 0,  sum(a) >= nu
% %
% %   There is no linear term, so the mass constraint is what makes the problem
% %   non-trivial: drop it and a = 0 is optimal. Each iteration projects onto
% %   {0 <= a <= up, y'a = 0} (Eq. 21) and, if the mass constraint is violated,
% %   re-projects with one multiplier per class so that each class carries nu/2
% %   (Eqs. 22-23).

%     n  = numel(y);
%     L  = ker.sig1;
%     up = P.up;
%     ip = (y > 0);
%     im = ~ip;

%     alpha        = zeros(n, 1);
%     z            = alpha;
%     tk           = 1;
%     ga           = zeros(n, 1);          % maintained K*alpha
%     gz           = zeros(n, 1);          % maintained K*z
%     sinceRefresh = 0;

%     hist = init_hist();
%     clk  = clk_new(P.setupPG);

%     it = 0;
%     while it < opts.maxIters
%         it = it + 1;

%         clk = clk_pause(clk);
%         % alpha = 0 at it == 1 violates sum(alpha) >= nu and has f = 0 < f*,
%         % which would wreck a log-suboptimality plot. Start recording once the
%         % iterate is feasible, i.e. after the first projection.
%         if it > 1 && mod(it - 1, opts.evalEvery) == 0
%             f    = plotted_objective(P, alpha, ga, y);
%             hist = rec_hist(hist, clk.solve, f);
%             maybe_print(opts, P.name, it, f);
%         end
%         stop = clk.solve >= opts.timeLimit;
%         clk  = clk_resume(clk);
%         if stop
%             break;
%         end

%         B = z - gz / L;

%         % ---- Eq. 21: project onto {0 <= a <= up, y'a = 0} -----------------
%         h   = @(lam) sum(y .* min(up, max(B - lam * y, 0)));
%         lam = bisect_root(h, max(1, max(abs(B))));
%         a   = min(up, max(B - lam * y, 0));

%         % ---- Eqs. 22-23: enforce nu/2 of mass in each class ---------------
%         if sum(a) < P.nu - 1e-12
%             hp = @(l) sum(min(up(ip), max(B(ip) - l, 0))) - P.nu / 2;
%             hm = @(l) sum(min(up(im), max(B(im) - l, 0))) - P.nu / 2;
%             lp = bisect_root(hp, max(1, max(abs(B(ip)))));
%             lm = bisect_root(hm, max(1, max(abs(B(im)))));

%             a(ip) = min(up(ip), max(B(ip) - lp, 0));
%             a(im) = min(up(im), max(B(im) - lm, 0));
%         end

%         dA        = a - alpha;
%         converged = norm(dA) <= opts.tol * max(1, norm(alpha));

%         theta = 0;
%         if opts.accel
%             if (z - a)' * dA > 0                 % gradient restart
%                 tk = 1;
%             else
%                 tk1   = (1 + sqrt(1 + 4 * tk ^ 2)) / 2;
%                 theta = (tk - 1) / tk1;
%                 tk    = tk1;
%             end
%         end
%         z = a + theta * dA;

%         sinceRefresh = sinceRefresh + 1;
%         gaPrev       = ga;

%         if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * n
%             ga = ga + ker.mul(sparse(dA));
%         else
%             ga           = ker.mul(a);
%             sinceRefresh = 0;
%         end
%         gz = ga + theta * (ga - gaPrev);

%         alpha = a;
%         if converged
%             break;
%         end
%     end

%     clk  = clk_pause(clk);
%     hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));

%     out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
% end


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
%   Timing. With -t 0 / -t 2 LIBSVM builds its own kernel cache inside svmtrain,
%   so the recorded time is exactly what it spends and nothing is pre-charged.
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

    if exist('svmtrain', 'file') ~= 3
        warning('LIBSVM mex (svmtrain) not on path; skipping SMO baseline.');
        return;
    end

    n       = size(X, 1);
    useKtil = false;

    if strcmp(opts.kernel, 'rbf')
        kpart = sprintf('-t 2 -g %.10g', opts.rbfGamma);   % matched gamma
    else
        kpart = '-t 0';
    end

    switch P.name
        case 'l1svm'
            base = sprintf('-s 0 %s -c %.10g', kpart, P.C);
            if strcmp(P.costMode, 'cost')
                wp   = P.Ci(find(y > 0, 1)) / P.C;
                wm   = P.Ci(find(y < 0, 1)) / P.C;
                base = sprintf('%s -w1 %.10g -w-1 %.10g', base, wp, wm);
            end

        case 'l2svm'
            if n > opts.explicitKernelMaxN
                warning(['l2svm SMO baseline needs the explicit %d x %d kernel ', ...
                         '(> opts.explicitKernelMaxN = %d); skipping.'], ...
                         n, n, opts.explicitKernelMaxN);
                return;
            end
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
        tKtr = tic;
        if strcmp(opts.kernel, 'rbf')
            Ktr = [(1:n)', rbf_gram(X, X, opts.rbfGamma) + diag(1 ./ P.Ci)];
        else
            Ktr = [(1:n)', full(X * X') + diag(1 ./ P.Ci)];
        end
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


function out = baseline_smo_mcsvm(ker, X, y, P, opts)
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

    if ker.explicit
        Kdiag = full(diag(ker.K));
    elseif strcmp(opts.kernel, 'rbf')
        Kdiag = ones(n, 1);                       % exp(0) = 1
    else
        Kdiag = full(sum(X .^ 2, 2));
    end
    Kdiag = max(Kdiag, 1e-12);

    alpha = zeros(n, K);
    KA    = zeros(n, K);                          % maintained K*alpha (= scores)
    epsB  = 1e-12;                                % box-activity tolerance

    clk  = clk_new(P.setupGram);                  % reads ker.K columns
    hist = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, KA, y));

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

        if ker.explicit
            Ki = ker.K(:, i);
        else
            Ki = ker.mul(sparse(i, 1, 1, n, 1));  % K(:, i) via one matvec
        end
        KA(:, u) = KA(:, u) + t * Ki;             % only two columns move
        KA(:, v) = KA(:, v) - t * Ki;

        % ---- record / time gate ------------------------------------------
        if mod(s, chkEvery) == 0
            clk  = clk_pause(clk);
            f    = plotted_objective(P, alpha, KA, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, 'smo-cs', s, f);
            stop = clk.solve >= opts.timeLimit;
            clk  = clk_resume(clk);
            if stop
                break;
            end
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, KA, y));

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

function out = baseline_dcd_binary(ker, X, y, P, opts)
%BASELINE_DCD_BINARY  Kernel SOR / kernel-adatron for L1-/L2-SVM.
%
%   Mangasarian & Musicant (1999); Friess et al. (1998). Equivalently, Hsieh et
%   al. (2008) Algorithm 3 -- random permutation plus shrinking -- with a kernel
%   gradient in place of the linear w-trick. (That trick maintains
%   w = sum_i y_i a_i x_i to read grad_i in O(nnz(x_i)); there is no w for an
%   RBF map, so a cached Gram column at O(n) is the best available.)
%
%   Dual (box-only; requires biasMode 'none' or 'augmented'):
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
                 'to ''none'' or ''augmented'' to put both solvers on the box-only ', ...
                 'dual. Skipping.']);
        return;
    end

    kacc = dcd_kernel_ops(ker, X, y, P, opts);
    if strcmp(kacc.mode, 'blocked')
        warning(['baseline_dcd_binary: RBF with no cached Gram. Every coordinate ', ...
                 'would need a fresh kernel column (O(nd)), making an epoch ', ...
                 'O(n^2 d) -- not a meaningful wall-clock number. Raise ', ...
                 'opts.explicitKernelMaxN above n = %d. Skipping.'], numel(y));
        return;
    end

    n      = numel(y);
    isL2   = strcmp(P.name, 'l2svm');
    linear = strcmp(kacc.mode, 'linear');

    if isL2
        U    = inf(n, 1);
        Qbar = kacc.diag + 1 ./ P.Ci;
    else
        U    = P.Ci;
        Qbar = kacc.diag;
    end

    alpha = zeros(n, 1);
    g     = zeros(n, 1);                 % g = Ktilde*alpha  (alpha = 0 -> 0)

    if linear
        Xa = X;
        if kacc.b2 > 0
            Xa = [X, opts.biasScale * ones(n, 1)];   % constant feature == the shift
        end
        w = zeros(size(Xa, 2), 1);       % w = Xa' * (y .* alpha)
    end

    % Shrinking state (Hsieh et al. Alg. 3).
    A    = (1:n)';
    Mbar =  inf;
    mbar = -inf;

    clk      = clk_new(P.setupGram);
    hist     = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, g, y));
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

            if linear
                gi = y(i) * (Xa(i, :) * w);
            else
                gi = g(i);
            end

            if isL2
                G = gi + alpha(i) / P.Ci(i) - 1;
            else
                G = gi - 1;
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
                    if linear
                        w = w + (delta * y(i)) * Xa(i, :)';   % O(nnz(x_i))
                    else
                        g = g + delta * kacc.col(i);          % O(n), cached column
                    end
                end
            end

            % ---- record / time gate --------------------------------------
            if mod(steps, chkEvery) == 0
                clk = clk_pause(clk);
                if linear
                    g = y .* (Xa * w);   % off-clock; keeps the objective exact
                end
                f    = plotted_objective(P, alpha, g, y);
                hist = rec_hist(hist, clk.solve, f);
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

    clk = clk_pause(clk);
    if linear
        g = y .* (Xa * w);
    end
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, g, y));

    out.alpha   = alpha;
    out.hist    = hist;
    out.skipped = false;
end


function out = baseline_dcd_svr(ker, X, y, P, opts)
%BASELINE_DCD_SVR  Coordinate descent for the eps-insensitive SVR dual.
%
%   Kernel form of Ho & Lin (2012) / LIBLINEAR -s 11..13.
%
%   Dual (box-only; requires biasMode 'none' or 'augmented'):
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
                 'opts.biasMode to ''none'' or ''augmented'' to compare on the ', ...
                 'box-only dual that coordinate descent actually solves. Skipping.']);
        return;
    end

    kacc = dcd_kernel_ops(ker, X, y, P, opts);
    if strcmp(kacc.mode, 'blocked')
        warning(['baseline_dcd_svr: RBF with no cached Gram -> O(n^2 d) per epoch. ', ...
                 'Raise opts.explicitKernelMaxN. Skipping.']);
        return;
    end

    n      = numel(y);
    epsIns = P.eps;
    linear = strcmp(kacc.mode, 'linear');
    Kii    = kacc.diag;

    b = zeros(n, 1);
    g = zeros(n, 1);                     % g = Ktilde*b

    if linear
        Xa = X;
        if kacc.b2 > 0
            Xa = [X, opts.biasScale * ones(n, 1)];
        end
        w = zeros(size(Xa, 2), 1);       % w = Xa' * b
    end

    A    = (1:n)';
    Mbar =  inf;
    mbar = -inf;

    clk      = clk_new(P.setupGram);
    hist     = rec_hist(hist, P.setupGram, plotted_objective(P, b, g, y));
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

            if linear
                gi = Xa(i, :) * w;
            else
                gi = g(i);
            end

            G  = gi - y(i);              % smooth part of the gradient
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
                    if linear
                        w = w + delta * Xa(i, :)';
                    else
                        g = g + delta * kacc.col(i);
                    end
                end
            end

            if mod(steps, chkEvery) == 0
                clk = clk_pause(clk);
                if linear
                    g = Xa * w;
                end
                f    = plotted_objective(P, b, g, y);
                hist = rec_hist(hist, clk.solve, f);
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

    clk = clk_pause(clk);
    if linear
        g = Xa * w;
    end
    hist = rec_hist(hist, clk.solve, plotted_objective(P, b, g, y));

    out.alpha   = b;
    out.hist    = hist;
    out.skipped = false;
end


function out = baseline_dcd_mcsvm(ker, X, y, P, opts)
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

    kacc = dcd_kernel_ops(ker, X, y, P, opts);
    if strcmp(kacc.mode, 'blocked')
        warning(['baseline_dcd_mcsvm: RBF with no cached Gram -> a kernel column ', ...
                 'per row. Raise opts.explicitKernelMaxN above n = %d. Skipping.'], ...
                 numel(y));
        return;
    end

    n = numel(y);
    K = P.K;

    E   = full(sparse((1:n)', y, 1, n, K));       % 1_ind
    CiK = repmat(P.Ci, 1, K);
    UP  = E .* CiK;                               % [0, C_i] at the true class
    LO  = (E - 1) .* CiK;                         % [-C_i, 0] elsewhere

    linear = strcmp(kacc.mode, 'linear');
    Kii    = kacc.diag;

    alpha = zeros(n, K);
    KA    = zeros(n, K);                          % maintained K*alpha
    if linear
        W = zeros(size(X, 2), K);                 % W = X'*alpha  (d x K)
    end

    A = (1:n)';                                   % active rows

    clk      = clk_new(P.setupGram);
    hist     = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, KA, y));
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

            if linear
                gi = X(i, :) * W;                 % 1 x K
            else
                gi = KA(i, :);
            end

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

                if linear
                    W  = W  + X(i, :)' * dRow;    % O(nnz(x_i) K)
                else
                    KA = KA + kacc.col(i) * dRow; % O(nK)
                end
            end

            if mod(steps, chkEvery) == 0
                clk = clk_pause(clk);
                if linear
                    KA = X * W;
                end
                f    = plotted_objective(P, alpha, KA, y);
                hist = rec_hist(hist, clk.solve, f);
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

    clk = clk_pause(clk);
    if linear
        KA = X * W;
    end
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, KA, y));

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
%   changing one a_i breaks its class sum. This is why augment_kernel_bias
%   refuses nusvm.
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

    % A pair step needs two kernel COLUMNS and the single ENTRY K_ij. Even a
    % linear kernel has no w-trick that yields K_ij cheaply, so a cached Gram is
    % required in every kernel mode.
    if ~ker.explicit
        warning(['baseline_pcd_nusvm: a pair step needs kernel columns and the ', ...
                 'entry K_ij, so a cached Gram is required. Raise ', ...
                 'opts.explicitKernelMaxN above n = %d. Skipping.'], numel(y));
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
    hist = rec_hist(hist, P.setupGram, plotted_objective(P, alpha, g, y));

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

        % ---- record / time gate ------------------------------------------
        if mod(s, chkEvery) == 0
            clk  = clk_pause(clk);
            f    = plotted_objective(P, alpha, g, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, 'pcd-nu', s, f);
            stop = clk.solve >= opts.timeLimit;
            clk  = clk_resume(clk);
            if stop
                break;
            end
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, g, y));

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
    hist = struct('t', [], 'f', []);
end


function hist = rec_hist(hist, t, f)
    hist.t(end+1, 1) = t;
    hist.f(end+1, 1) = f;
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
