function results = run_svm_comparison(matFile, opts)
    addpath('./libsvm-336/matlab');
    addpath('./liblinear-249/matlab');
%RUN_SVM_COMPARISON Dual projected-gradient SVM solvers vs SMO baselines.
% run_svm_comparison('dataset.mat');
% output: <outRoot>/<dataset>/<problem>_<costMode>.png

    % ========== %
    %   THEORY   %
    % ========== %

%===%  L1/L2 SVM ('l1svm' 'l2svm')
% Equation 1: min_(w,b,e) .5*||w||^2_2 + C*sum^n_i e_i, s.t. y_i(w'*x_i + b) >= 1 - e_i, e_i >= 0, i=1,...,n
% Equation 2: max_alpha sum^n_i alpha_i - .5*sum^n_i sum^n_j alpha_i * alpha_j * y_i * y_j * x'_i * x_j, s.t. 0 <= alpha_i <= C, sum^n_i alpha_i * y_i = 0
% Equation 5: min_alpha f(alpha) = .5 alpha'*K*alpha + 1/(2*C) ||alpha||^2_2 - alpha'*1, s.t. alpha >= 0, <alpha, y> = 0.
%
% L2-SVM or L1-SVM:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K) + 1/C or sigma_1(K), initalize alpha = 0.
% Ensure approximate solution alpha to Eq 5.
% while not converged do:
%         beta <- alpha - 1/L(K*alpha+alpha/C-1) or alpha - 1/L(K*alpha -1)
%         find lamdba: sum^n_i y_i[beta_i - lambda*y_i]_+ or sum^n_i y_i min{C, max{beta_i - lambda*y_i,0}} = 0 via bisection
%         alpha <- [beta - lambda*]_+ or min{C, max{beta - lambda*y,0}}
% end while

%===% SVR ('svr')
% Equation 10: min_beta .5*beta'*K*beta - y'*beta + epsilon||beta||_1, s.t. 1'*beta = 0, ||beta||_inf <= C.
%
% SVR:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K), initalize beta = 0.
% Ensure: optimial beta to Eq 10
% while not converged do
%         v <- beta - 1/L(K*beta-y)
%         lambda: sum^n_i clip(S_{epsilon/L}(v_i - lambda), -C, C)=0 via bisection
%         beta <- clip(S_{epsilon/L}(v-lambda*1), -C, C)
% end while

%===% Multiclass SVM ('mcsvm')
% Equation 16: min{C, max{0,b^(t)_(i,y_i) - lambda_i}} + sum_(j!=y_i) min{0, max{-C, b^(t)_(i,j) - lambda_i}} = 0
%
% Jacobi-style parallel update for multiclass SVM:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K), initalize beta = 0.
% for t = 0,1,2,... until convergence do
%     B^(t) <- alpha^(t) - 1/L(K*alpha^(t) - 1_ind)
%     for i = 1,...,n in parallel
%         Find lambda_i satisfying Eq 16 using bisection
%         alpha^(t+1)_(i,y_i) <- min{C, max{0,b^(t)_(i,y_i) - lambda_i}}, alpha^(t)_(i,j) - lambda_i}{j!=y_i}
%     end for
% end for

%===% Nu-SVM ('nusvm')
% Equation 21: sum^n_i y_i min{1/n, max{0, beta_i^(t) - lambda*y_i}} = 0
% Equation 22: alpha_i = min{1/n, max{0, beta_i^(t) - lambda_+}}, y_i = +1; alpha_i = min{1/n, max{0, beta^(t)_i - lambda_-}}, y_i = -1;
% Equation 23: sum_(i:y_i=+1) min{1/n, max{0, beta_i^(t) - lambda_+}} = nu/2; sum_(i:y_i=-1) min{1/n, max{0, beta^(t)_i - lambda_-}} = vu/2;
%
% nu-SVM:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K), initalize beta = 0.
% for t = 0,1,2,... until convergence do
%     B^(t) <- alpha^(t) - 1/L*K*alpha^(t)
%     alpha_i <- min{1/n, max{0, beta_i^(t) - lambda*y_i}}
%     if sum^n_i alpha_i < nu then
%         solve Eq 23 using bisection
%         Update alpha via Eq 22
%     end if
%     alpha^(t+1) <- alpha
% end for

    % ========== %
    % END THEORY %
    % ========== %

% Libsvm Equivalent Methods:
%   l1svm : LIBSVM  -s 0 -t 0 (exact same dual; -wi class costs)
%   l2svm : LIBSVM  -s 0 -t 4 with precomputed kernel K + diag(1./C_i)
%           and a huge box: the classic reduction of L2-SVM to
%           hard-margin SMO. Needs the explicit n x n kernel, so it runs
%           only when n <= opts.explicitKernelMaxN.
%   nusvm : LIBSVM  -s 1 -t 0 (alpha rescaled so sum = nu)
%   svr   : LIBSVM  -s 3 -t 0
%   mcsvm : LIBLINEAR -s 4 (Crammer-Singer dual coordinate descent)
%
% Cost-sensitive mode: (opts.costMode = 'cost') vs (opts.costMode = 'none')
%  - 'classification': C_i = C * w_{y_i};
%  - 'nusvm': up_i = w_{y_i}/n
%  - 'svr': [-C*costNegFactor, +C*costPosFactor]
%
% Kernel: RBF with radius 1
%
% Dataset/task augments:
%   binary problems on multiclass data : opts.binarize (default halfsplit)
%   binary problems on regression data : median split of y      [EDIT-ME]
%   mcsvm on regression data           : opts.mcBins quantile bins (4)
%   svr on classification data         : regress on the +/-1 labels
%                                        (optimization benchmark) [EDIT-ME]

    if nargin < 2
        opts = struct();
    end
    opts = fill_default_opts(opts);
    rng(opts.seed);

    [~, datasetName] = fileparts(matFile);
    figDir = fullfile(opts.outRoot, datasetName);
    figPath = fullfile(figDir, sprintf('%s_%s.png', opts.problem, opts.costMode));

    if exist(figPath, 'file') && ~opts.overwrite
        fprintf('[skip] %s exists (set opts.overwrite = true to redo)\n', figPath);
        results = struct('skipped', true, 'figPath', figPath);
        return;
    end
    if ~exist(figDir, 'dir')
        mkdir(figDir);
    end

    [X, y] = load_xy_from_mat(matFile);
    [X, y, meta] = preprocess_xy(X, y, opts);
    P = make_problem(y, meta, opts);

    n = size(X, 1);
    fprintf('%s | n = %d, d = %d | problem = %s | costMode = %s | C = %g\n', ...
        datasetName, n, size(X, 2), opts.problem, opts.costMode, opts.C);

    ker = make_kernel_op(X, y, P, opts);
    % P.chargedSetup = ker.setupTime;   % sigma_1 / K build time billed to us
    P.chargedSetup = 0;
    fprintf('sigma_1(K) = %.4e (setup %.2f s, charged to the proposed method)\n', ...
        ker.sig1, ker.setupTime);

    % ---- proposed method -------------------------------------------------
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

    % ---- baseline --------------------------------------------------------
    if strcmp(P.name, 'mcsvm')
        baseOut   = baseline_smo_mcsvm(ker, X, y, P, opts);   % SMO family
        baseLabel = 'CS SMO (max-violating pair, analytic step)';
    else
        baseOut = baseline_libsvm_sweep(ker, X, y, P, opts);
        baseLabel = 'LIBSVM SMO (tol sweep)';
    end

    % ---- figure (the only artifact) --------------------------------------
    hists = {propOut.hist};
    labels = {propLabel};
    if ~baseOut.skipped
        hists{end+1} = baseOut.hist;
        labels{end+1} = baseLabel;
    end
    ylab = 'Dual suboptimality  f - f*';
    ttl = sprintf('%s: %s (%s)', datasetName, P.name, opts.costMode);
    if opts.makeFigure
        make_config_figure(hists, labels, figPath, ttl, ylab);
        fprintf('Saved %s\n', figPath);
    end

    results = struct();
    results.skipped = false;
    results.figPath = figPath;
    results.problem = P.name;
    results.costMode = opts.costMode;
    results.proposed = propOut;
    results.baseline = baseOut;
    results.labels = {labels};
    results.opts = opts;
end


% ======================================================================
% Options
% ======================================================================

function opts = fill_default_opts(opts)
    opts = set_default(opts, 'problem', 'nusvm'); %l1svm l2svm svr nusvm mcsvm
    opts = set_default(opts, 'costMode', 'none');
    opts = set_default(opts, 'kernel', 'rbf'); % 'linear' | 'rbf'
    opts = set_default(opts, 'rbfGamma', 0.5); % k = exp(-g ||x-x'||^2);
                                               % g = 1/(2 r^2), radius 1
    opts = set_default(opts, 'accel', true);
    opts = set_default(opts, 'lazy', true);       % incremental K*delta updates
    opts = set_default(opts, 'lazyRefresh', 100); % full recompute cadence
                                                  % (guards FP drift)

    opts = set_default(opts, 'C', 1);
    opts = set_default(opts, 'nu', 0.2);
    opts = set_default(opts, 'epsSVR', 0.1);
    opts = set_default(opts, 'classCosts', []); % per-class multipliers
    opts = set_default(opts, 'costPosFactor', 2); % svr 'cost': over-est box
    opts = set_default(opts, 'costNegFactor', 1); % svr 'cost': under-est box
    opts = set_default(opts, 'task', '');
    opts = set_default(opts, 'binarize', struct('type', 'halfsplit'));
    opts = set_default(opts, 'mcBins', 4);
    opts = set_default(opts, 'maxIters', 5000);
    opts = set_default(opts, 'tol', 1e-10);
    opts = set_default(opts, 'timeLimit', 60);
    opts = set_default(opts, 'evalEvery', 10);
    opts = set_default(opts, 'printEvery', 200);
    opts = set_default(opts, 'maxSamples', inf);
    opts = set_default(opts, 'standardize', true);
    opts = set_default(opts, 'standardizeY', true);
    opts = set_default(opts, 'explicitKernelMaxN', 80000);
    opts = set_default(opts, 'smoTolerances', 10.^-(1:8));
    opts = set_default(opts, 'liblinearFn', 'train');
    opts = set_default(opts, 'outRoot', '.');
    opts = set_default(opts, 'overwrite', false);
    opts = set_default(opts, 'seed', 1);
    opts = set_default(opts, 'verbose', true);
    opts = set_default(opts, 'makeFigure', true);

    valid = {'l1svm', 'l2svm', 'nusvm', 'mcsvm', 'svr'};
    if ~any(strcmp(opts.problem, valid))
        error('opts.problem must be one of: %s', strjoin(valid, ', '));
    end
    valid = {'none', 'cost'};
    if ~any(strcmp(opts.costMode, valid))
        error('opts.costMode must be ''none'' or ''cost''.');
    end
end


function opts = set_default(opts, name, value)
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = value;
    end
end


% ======================================================================
% Data loading, task adaptation, costs
% ======================================================================

function [X, y] = load_xy_from_mat(matFile)
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
        if isfield(S, nm)
            yy = S.(nm);
            if (isnumeric(yy) || islogical(yy)) && isvector(yy)
                y = yy;
                break;
            elseif isnumeric(yy) && ismatrix(yy) && size(yy,1) == size(X,1) && size(yy,2) > 1
                [~, y] = max(yy, [], 2);   % one-hot -> labels
                break;
            end
        end
    end

    if isempty(X) || isempty(y)
        names = fieldnames(S);
        for i = 1:numel(names)
            A = S.(names{i});
            if (isnumeric(A) || islogical(A)) && ismatrix(A) && ~isvector(A)
                for j = 1:numel(names)
                    b = S.(names{j});
                    if (isnumeric(b) || islogical(b)) && isvector(b) && numel(b) == size(A,1)
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
        error(['Could not identify X and y in %s. ', ...
               'Store variables as X and y, or edit load_xy_from_mat().'], matFile);
    end

    if ~issparse(X)
        X = double(X);
    end
    y = double(y(:));
end


function task = detect_task(y)
    isInt = all(abs(y - round(y)) < 1e-9);
    u = unique(y);
    if isInt && numel(u) <= 2000 && numel(u) <= numel(y) / 2
        if numel(u) == 2
            task = 'binary';
        else
            task = 'multiclass';
        end
    else
        task = 'regression';
    end
    fprintf('Task not specified; inferred "%s" (%d unique targets). Override with opts.task.\n', ...
        task, numel(u));
end


function [X, y, meta] = preprocess_xy(X, y, opts)
    if size(X,1) ~= numel(y) && size(X,2) == numel(y)
        X = X';
    end
    if size(X,1) ~= numel(y)
        error('X and y dimensions do not match.');
    end

    if isempty(opts.task)
        task = detect_task(y);
    else
        task = opts.task;
    end

    n = size(X,1);
    if isfinite(opts.maxSamples) && n > opts.maxSamples
        N = round(opts.maxSamples);
        sel = randperm(n, N);
        X = X(sel, :);
        y = y(sel);
        fprintf('Subsampled %d -> %d rows.\n', n, N);
    end

    % ---- adapt targets to the requested problem --------------------------
    isBinaryProblem = any(strcmp(opts.problem, {'l1svm', 'l2svm', 'nusvm'}));
    K = 1;
    if isBinaryProblem
        switch task
            case 'binary'
                [~, ~, yi] = unique(y);
                y = 2*(yi == 2) - 1;
            case 'multiclass'
                [~, ~, yi] = unique(y);
                y = binarize_labels(yi, opts.binarize, max(yi));
            case 'regression'
                y = 2*(y > median(y)) - 1;
                fprintf('Regression targets median-split into +/-1 labels.\n');
        end
        K = 2;
    elseif strcmp(opts.problem, 'mcsvm')
        if strcmp(task, 'regression')
            edges = quantile(y, (1:opts.mcBins-1) / opts.mcBins);
            yb = ones(numel(y), 1);
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
    else % svr
        if ~strcmp(task, 'regression')
            [~, ~, yi] = unique(y);
            if max(yi) == 2
                y = 2*(yi == 2) - 1;
            else
                y = binarize_labels(yi, opts.binarize, max(yi));
            end
            fprintf('Classification labels used as +/-1 regression targets for svr.\n');
        elseif opts.standardizeY
            y = (y - mean(y)) / max(std(y), 1e-12);  % makes epsSVR comparable
        end
    end

    if opts.standardize
        if issparse(X)
            colScale = full(sqrt(sum(X.^2, 1) / max(1, size(X,1))));
            colScale(colScale < 1e-12) = 1;
            X = X * spdiags(1 ./ colScale(:), 0, size(X,2), size(X,2));
        else
            mu = mean(X, 1);
            sigma = std(X, 0, 1);
            sigma(sigma < 1e-12) = 1;
            X = bsxfun(@rdivide, bsxfun(@minus, X, mu), sigma);
        end
    end

    meta = struct('task', task, 'K', K);
end


function y = binarize_labels(y, spec, K)
    switch spec.type
        case 'halfsplit'
            y = 2*(y <= floor(K/2)) - 1;
        case 'ovr'
            y = 2*(y == spec.k) - 1;
        otherwise
            error('Unknown binarize type "%s".', spec.type);
    end
    if all(y == y(1))
        error('Binarization produced a single class; adjust opts.binarize.');
    end
end


function P = make_problem(y, meta, opts)
    n = numel(y);
    P = struct();
    P.name = opts.problem;
    P.costMode = opts.costMode;
    P.C = opts.C;
    P.nu = opts.nu;
    P.eps = opts.epsSVR;
    P.K = meta.K;
    P.chargedSetup = 0;

    withCost = strcmp(opts.costMode, 'cost');

    switch P.name
        case {'l1svm', 'l2svm'}
            w = class_weights_pm(y, withCost, opts);
            P.Ci = opts.C * w; % per-sample box / diag
        case 'nusvm'
            w = class_weights_pm(y, withCost, opts);
            P.up = w / n; % Eq 21-23 box
            sp = sum(P.up(y > 0));
            sm = sum(P.up(y < 0));
            if opts.nu/2 > min(sp, sm) + 1e-12
                error(['nu = %g infeasible: per-class box mass is (%.4g, %.4g) ', ...
                       'but Eq 23 needs >= nu/2 = %.4g each. Reduce nu.'], ...
                       opts.nu, sp, sm, opts.nu/2);
            end
        case 'mcsvm'
            if withCost
                if isempty(opts.classCosts)
                    counts = accumarray(y, 1, [meta.K, 1]);
                    w = n ./ (meta.K * counts); % balanced default
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
    n = numel(y);
    if ~withCost
        w = ones(n, 1);
        return;
    end
    if ~isempty(opts.classCosts)
        cc = opts.classCosts(:);
        w = cc(1) * ones(n, 1);
        w(y > 0) = cc(2);
    else
        np = sum(y > 0);
        nm = n - np;
        w = ones(n, 1);
        w(y > 0) = n / (2 * np);
        w(y < 0) = n / (2 * nm);
    end
end


% ======================================================================
% Kernel
% ======================================================================
function ker = make_kernel_op(X, y, P, opts)
    n = size(X, 1);
    signed = any(strcmp(P.name, {'l1svm', 'l2svm', 'nusvm'}));
    isRbf = strcmp(opts.kernel, 'rbf');
    tSetup = tic;

    if n <= opts.explicitKernelMaxN
        if isRbf
            K = rbf_gram(X, X, opts.rbfGamma);
        else
            K = full(X * X');
        end
        if signed
            K = (y * y') .* K;
        end
        ker.mul = @(a) K * a;
        ker.explicit = true;
        ker.K = K;
    elseif isRbf
        % O(n^2 d) per iteration, large samples datasets slow
        warning('n = %d > explicitKernelMaxN: RBF K*a computed in blocks (slow).', n);
        if signed
            ker.mul = @(a) y .* rbf_mul_blocked(X, opts.rbfGamma, y .* a);
        else
            ker.mul = @(a) rbf_mul_blocked(X, opts.rbfGamma, a);
        end
        ker.explicit = false;
    else
        Xt = X'; % cache the transpose once
        if signed
            ker.mul = @(a) y .* (X * (Xt * (y .* a)));
        else
            ker.mul = @(a) X * (Xt * a); % works for n x K matrices too
        end
        ker.explicit = false;
    end

    if isRbf
        v = randn(n, 1);
        v = v / norm(v);
        for k = 1:100
            v = ker.mul(v);
            v = v / max(norm(v), 1e-300);
        end
        ker.sig1 = min(1.05 * (v' * ker.mul(v)), n);
    else
        % sigma_1(K) = sigma_max(X)^2 exactly for the linear kernel.
        try
            ker.sig1 = svds(X, 1)^2;
        catch
            v = randn(n, 1);
            for k = 1:50
                v = ker.mul(v);
                v = v / max(norm(v), 1e-300);
            end
            ker.sig1 = 1.05 * (v' * ker.mul(v));
        end
    end
    ker.sig1 = max(ker.sig1, 1e-12);
    ker.setupTime = toc(tSetup);
end


function K = rbf_gram(A, B, gamma)
% k(a, b) = exp(-gamma ||a - b||^2). gamma = 1/(2 radius^2) matches
% LIBSVM's -g convention, so radius 1 <=> gamma 0.5.
    sqA = full(sum(A.^2, 2));
    sqB = full(sum(B.^2, 2));
    D2 = bsxfun(@plus, sqA, sqB') - 2 * full(A * B');
    K = exp(-gamma * max(D2, 0));      % clamp tiny negative round-off
end


function out = rbf_mul_blocked(X, gamma, a)
% K*a for the RBF kernel without materializing K: rows in blocks sized to
% roughly 1 GB of doubles. Accepts n x 1 or n x K right-hand sides.
% Sparse a (lazy delta updates) hits a fast path: the Gram is formed only
% against the support of a, so cost is O(n |supp| d) instead of O(n^2 d).
    n = size(X, 1);
    if issparse(a)
        idx = find(any(a ~= 0, 2));
        if isempty(idx)
            out = zeros(n, size(a, 2));
            return;
        end
        Xs = X(idx, :);
        af = full(a(idx, :));
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
        idx = i0:min(i0 + blk - 1, n);
        out(idx, :) = rbf_gram(X(idx, :), X, gamma) * a;
    end
end

% ======================================================================
% Objective value evaluation
% ======================================================================

function f = plotted_objective(P, a, g, y)
    switch P.name
        case 'l1svm'
            f = 0.5 * (a' * g) - sum(a);
        case 'l2svm'
            f = 0.5 * (a' * g) + 0.5 * sum(a.^2 ./ P.Ci) - sum(a);
        case 'nusvm'
            f = 0.5 * (a' * g);
        case 'svr'
            f = 0.5 * (a' * g) - y' * a + P.eps * sum(abs(a));
        case 'mcsvm'
        % DUAL objective d(a) = 1/2 <a, K a> - <E, a>. This is exactly the
        % function solve_mcsvm and baseline_smo_mcsvm minimize (grad = K a - E).
        % Both maintain a and g = K a, so it's free and shares d*.
        n = numel(y);
        idxTrue = sub2ind(size(a), (1:n)', y);
        f = 0.5 * sum(sum(a .* g)) - sum(a(idxTrue));
        % --- primal (previous behavior), kept for reference ---
        % Drow = 1 - full(sparse((1:n)', y, 1, n, P.K));
        % loss = max(Drow + g, [], 2) - g(idxTrue);
        % f = 0.5 * sum(sum(a .* g)) + sum(P.Ci .* loss);
    end
end


% ======================================================================
% Solvers
% ======================================================================

%====% L1 & L2 SVM %====%
% Equation 5: min_alpha f(alpha) = .5 alpha'*K*alpha + 1/(2*C) ||alpha||^2_2 - alpha'*1, s.t. alpha >= 0, <alpha, y> = 0.
%
% L2-SVM or L1-SVM:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K) + 1/C or sigma_1(K), initalize alpha = 0.
% Ensure approximate solution alpha to Eq 5.
% while not converged do:
%         beta <- alpha - 1/L(K*alpha+alpha/C-1) or alpha - 1/L(K*alpha -1)
%         find lamdba: sum^n_i y_i[beta_i - lambda*y_i]_+ or sum^n_i y_i min{C, max{beta_i - lambda*y_i,0}} = 0 via bisection
%         alpha <- [beta - lambda*]_+ or min{C, max{beta - lambda*y,0}}
% end while
function out = solve_l1l2(ker, y, P, opts)
    n = numel(y);
    isL2 = strcmp(P.name, 'l2svm');
    if isL2
        L = ker.sig1 + 1 / min(P.Ci);
    else
        L = ker.sig1;
    end

    alpha = zeros(n, 1);
    z = alpha;
    tk = 1;
    ga = zeros(n, 1); % maintained K*alpha (alpha = 0 -> 0)
    gz = zeros(n, 1); % maintained K*z
    sinceRefresh = 0;
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        if isL2
            grad = gz + z ./ P.Ci - 1;
        else
            grad = gz - 1;
        end

        clk = clk_pause(clk); % evaluation off-clock
        if mod(it - 1, opts.evalEvery) == 0
            f = plotted_objective(P, alpha, ga, y); % free: ga is maintained
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        beta = z - grad / L;
        scale = max(1, max(abs(beta)));
        if isL2
            h = @(lam) sum(y .* max(beta - lam * y, 0));
        else
            h = @(lam) sum(y .* min(P.Ci, max(beta - lam * y, 0)));
        end
        lam = bisect_root(h, scale);
        if isL2
            alphaNew = max(beta - lam * y, 0);
        else
            alphaNew = min(P.Ci, max(beta - lam * y, 0));
        end

        dA = alphaNew - alpha;
        converged = norm(dA) <= opts.tol * max(1, norm(alpha));

        if isL2 % Stongly convex
            mu = 1 / max(P.Ci);
        else % Not strongly convex
            mu = 0;
        end
        [theta, tk] = momentum_step(opts, mu, L, tk, (z - alphaNew)' * dA);
        [theta, tk] = momentum_step(opts, mu, L, tk, (z - alphaNew)' * dA);
        z = alphaNew + theta * dA;

        % theta = 0;
        % if opts.accel
        %     if (z - alphaNew)' * dA > 0
        %         tk = 1;
        %     else
        %         tk1 = (1 + sqrt(1 + 4 * tk^2)) / 2;
        %         theta = (tk - 1) / tk1;
        %         tk = tk1;
        %     end
        % end
        % z = alphaNew + theta * dA;

        % Lazy kernel-image maintenance: bound coordinates have dA_i = 0
        % exactly (clip -> clip), so late-phase deltas are supported on
        % the free SVs only and K*dA is a cheap sparse matvec. z is a
        % linear combination of the last two alphas, so K*z falls out of
        % the two maintained images with no extra matvec.
        sinceRefresh = sinceRefresh + 1;
        gaPrev = ga;
        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * n
            ga = ga + ker.mul(sparse(dA));
        else
            ga = ker.mul(alphaNew); % periodic full refresh
            sinceRefresh = 0;
        end
        gz = ga + theta * (ga - gaPrev);

        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end

%====% SVR %====%
% Equation 10: min_beta .5*beta'*K*beta - y'*beta + epsilon||beta||_1, s.t. 1'*beta = 0, ||beta||_inf <= C.
%
% SVR:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K), initalize beta = 0.
% Ensure: optimial beta to Eq 10
% while not converged do
%         v <- beta - 1/L(K*beta-y)
%         lambda: sum^n_i clip(S_{epsilon/L}(v_i - lambda), -C, C)=0 via bisection
%         beta <- clip(S_{epsilon/L}(v-lambda*1), -C, C)
% end while
function out = solve_svr(ker, y, P, opts)
    n = numel(y);
    L = ker.sig1;
    thr = P.eps / L;
    st = @(u) sign(u) .* max(abs(u) - thr, 0);   % soft threshold S_{eps/L}

    b = zeros(n, 1);
    z = b;
    tk = 1;
    gb = zeros(n, 1); % maintained K*b
    gz = zeros(n, 1); % maintained K*z
    sinceRefresh = 0;
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f = plotted_objective(P, b, gb, y); % free: gb is maintained
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        v = z - (gz - y) / L;
        h = @(lam) sum(min(P.bUp, max(P.bLo, st(v - lam))));
        lam = bisect_root(h, max(1, max(abs(v))));
        bNew = min(P.bUp, max(P.bLo, st(v - lam)));

        dB = bNew - b;
        converged = norm(dB) <= opts.tol * max(1, norm(b));

        theta = 0;
        if opts.accel
            if (z - bNew)' * dB > 0 % gradient restart
                tk = 1;
            else
                tk1 = (1 + sqrt(1 + 4 * tk^2)) / 2;
                theta = (tk - 1) / tk1;
                tk = tk1;
            end
        end
        z = bNew + theta * dB;

        sinceRefresh = sinceRefresh + 1;
        gbPrev = gb;
        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dB) < 0.5 * n
            gb = gb + ker.mul(sparse(dB));
        else
            gb = ker.mul(bNew);
            sinceRefresh = 0;
        end
        gz = gb + theta * (gb - gbPrev);

        b = bNew;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, b, ker.mul(b), y));

    out = struct('alpha', b, 'hist', hist, 'iters', it, 'skipped', false);
end

%====% Multiclass SVM %====%
% Equation 16: min{C, max{0,b^(t)_(i,y_i) - lambda_i}} + sum_(j!=y_i) min{0, max{-C, b^(t)_(i,j) - lambda_i}} = 0
%
% Jacobi-style parallel update for multiclass SVM:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K), initalize beta = 0.
% for t = 0,1,2,... until convergence do
%     B^(t) <- alpha^(t) - 1/L(K*alpha^(t) - 1_ind)
%     for i = 1,...,n in parallel
%         Find lambda_i satisfying Eq 16 using bisection
%         alpha^(t+1)_(i,y_i) <- min{C, max{0,b^(t)_(i,y_i) - lambda_i}}, alpha^(t)_(i,j) - lambda_i}{j!=y_i}
%     end for
% end for
function out = solve_mcsvm(ker, y, P, opts)
    n = numel(y);
    K = P.K;
    L = ker.sig1;

    E = full(sparse((1:n)', y, 1, n, K)); % 1_ind
    CiK = repmat(P.Ci, 1, K);
    UP = E .* CiK; % [0, C_i] at true class
    LO = (E - 1) .* CiK; % [-C_i, 0] elsewhere

    alpha = zeros(n, K);
    Z = alpha;
    tk = 1;
    GA = zeros(n, K); % maintained K*alpha
    GZ = zeros(n, K); % maintained K*Z
    sinceRefresh = 0;
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            % f = plotted_objective(P, alpha, ker.mul(alpha), y);
            f = plotted_objective(P, alpha, GA, y); % free: GA is maintained
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        B = Z - (GZ - E) / L;

        % Eq 16 for every row simultaneously: each h_i is nonincreasing in
        % lambda_i, positive at lo and nonpositive at hi, so a vectorized
        % bisection runs all n root-finds in parallel.
        lo = min(B, [], 2) - max(P.Ci) - 1;
        hi = max(B, [], 2) + 1;
        for k = 1:80
            mid = (lo + hi) / 2;
            Cl = min(UP, max(LO, bsxfun(@minus, B, mid)));
            pos = sum(Cl, 2) > 0;
            lo(pos) = mid(pos);
            hi(~pos) = mid(~pos);
        end
        mid = (lo + hi) / 2;
        alphaNew = min(UP, max(LO, bsxfun(@minus, B, mid)));

        dA = alphaNew - alpha;
        converged = norm(dA(:)) <= opts.tol * max(1, norm(alpha(:)));

        theta = 0;
        if opts.accel
            if sum(sum((Z - alphaNew) .* dA)) > 0 % gradient restart
                tk = 1;
            else
                tk1 = (1 + sqrt(1 + 4 * tk^2)) / 2;
                theta = (tk - 1) / tk1;
                tk = tk1;
            end
        end
        Z = alphaNew + theta * dA;

        % Lazy maintenance: rows whose every coordinate stayed clipped at
        % its bound have an all-zero delta row, so K*dA touches only the
        % active rows (sparse fast paths in all three kernel modes).
        sinceRefresh = sinceRefresh + 1;
        GAPrev = GA;
        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * numel(dA)
            GA = GA + ker.mul(sparse(dA));
        else
            GA = ker.mul(alphaNew);
            sinceRefresh = 0;
        end
        GZ = GA + theta * (GA - GAPrev);

        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));


    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end

%====% Nu-SVM %====%
% Equation 21: sum^n_i y_i min{1/n, max{0, beta_i^(t) - lambda*y_i}} = 0
% Equation 22: alpha_i = min{1/n, max{0, beta_i^(t) - lambda_+}}, y_i = +1; alpha_i = min{1/n, max{0, beta^(t)_i - lambda_-}}, y_i = -1;
% Equation 23: sum_(i:y_i=+1) min{1/n, max{0, beta_i^(t) - lambda_+}} = nu/2; sum_(i:y_i=-1) min{1/n, max{0, beta^(t)_i - lambda_-}} = nu/2;
%
% nu-SVM:
% Given C, (x_i, y_i)^n_i, formulate K, L = sigma_1(K), initalize beta = 0.
% for t = 0,1,2,... until convergence do
%     B^(t) <- alpha^(t) - 1/L*K*alpha^(t)
%     alpha_i <- min{1/n, max{0, beta_i^(t) - lambda*y_i}}
%     if sum^n_i alpha_i < nu then
%         solve Eq 23 using bisection
%         Update alpha via Eq 22
%     end if
%     alpha^(t+1) <- alpha
% end for
function out = solve_nusvm(ker, y, P, opts)
    n = numel(y);
    L = ker.sig1;
    up = P.up;
    ip = (y > 0);
    im = ~ip;

    alpha = zeros(n, 1);
    z = alpha;
    tk = 1;
    ga = zeros(n, 1); % maintained K*alpha
    gz = zeros(n, 1); % maintained K*z
    sinceRefresh = 0;
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        clk = clk_pause(clk);
        % alpha = 0 (it == 1) violates sum(alpha) >= nu and has f = 0 < f*,
        % which would wreck a log suboptimality plot, so recording starts
        % once the iterate is feasible (after the first projection).
        if it > 1 && mod(it - 1, opts.evalEvery) == 0
            f = plotted_objective(P, alpha, ga, y); % free: ga is maintained
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        B = z - gz / L;

        % Eq 21: project onto {0 <= a <= up, y'a = 0}
        h = @(lam) sum(y .* min(up, max(B - lam * y, 0)));
        lam = bisect_root(h, max(1, max(abs(B))));
        a = min(up, max(B - lam * y, 0));

        % Eq 22-23: if the mass constraint is violated, enforce
        % sum_{+} a = sum_{-} a = nu/2 with one multiplier per class.
        if sum(a) < P.nu - 1e-12
            hp = @(l) sum(min(up(ip), max(B(ip) - l, 0))) - P.nu / 2;
            hm = @(l) sum(min(up(im), max(B(im) - l, 0))) - P.nu / 2;
            lp = bisect_root(hp, max(1, max(abs(B(ip)))));
            lm = bisect_root(hm, max(1, max(abs(B(im)))));
            a(ip) = min(up(ip), max(B(ip) - lp, 0));
            a(im) = min(up(im), max(B(im) - lm, 0));
        end

        dA = a - alpha;
        converged = norm(dA) <= opts.tol * max(1, norm(alpha));

        theta = 0;
        if opts.accel
            if (z - a)' * dA > 0 % gradient restart
                tk = 1;
            else
                tk1 = (1 + sqrt(1 + 4 * tk^2)) / 2;
                theta = (tk - 1) / tk1;
                tk = tk1;
            end
        end
        z = a + theta * dA;

        sinceRefresh = sinceRefresh + 1;
        gaPrev = ga;
        if opts.lazy && sinceRefresh < opts.lazyRefresh && nnz(dA) < 0.5 * n
            ga = ga + ker.mul(sparse(dA));
        else
            ga = ker.mul(a);
            sinceRefresh = 0;
        end
        gz = ga + theta * (ga - gaPrev);

        alpha = a;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    g = ker.mul(alpha);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, ker.mul(alpha), y));

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end

function out = baseline_libsvm_sweep(ker, X, y, P, opts)
    hist = init_hist();
    out = struct('alpha', [], 'hist', hist, 'skipped', true);

    if exist('svmtrain', 'file') ~= 3
        warning('LIBSVM mex (svmtrain) not on path; skipping SMO baseline.');
        return;
    end

    n = size(X, 1);
    useKtil = false;

    if strcmp(opts.kernel, 'rbf')
        kpart = sprintf('-t 2 -g %.10g', opts.rbfGamma); % LIBSVM RBF, matched gamma
    else
        kpart = '-t 0'; % linear
    end

    switch P.name
        case 'l1svm'
            base = sprintf('-s 0 %s -c %.10g', kpart, P.C);
            if strcmp(P.costMode, 'cost')
                wp = P.Ci(find(y > 0, 1)) / P.C;
                wm = P.Ci(find(y < 0, 1)) / P.C;
                base = sprintf('%s -w1 %.10g -w-1 %.10g', base, wp, wm);
            end
        case 'l2svm'
            if n > opts.explicitKernelMaxN
                warning(['l2svm SMO baseline needs the explicit %d x %d kernel ', ...
                         '(> opts.explicitKernelMaxN = %d); skipping.'], ...
                         n, n, opts.explicitKernelMaxN);
                return;
            end
            bigC = 1e10 * max(P.Ci);
            base = sprintf('-s 0 -t 4 -c %.10g', bigC);
            useKtil = true;
        case 'nusvm'
            if strcmp(P.costMode, 'cost')
                warning('Stock LIBSVM has no weighted nu-SVC; skipping baseline.');
                return;
            end
            base = sprintf('-s 1 %s -n %.10g', kpart, P.nu);
        case 'svr'
            if strcmp(P.costMode, 'cost')
                warning('Stock LIBSVM eps-SVR has one C (no asymmetric box); skipping baseline.');
                return;
            end
            base = sprintf('-s 3 %s -c %.10g -p %.10g', kpart, P.C, P.eps);
        otherwise
            return;
    end

    % One-off conversions, excluded from the recorded training times.
    if useKtil
        if strcmp(opts.kernel, 'rbf')
            Ktr = [(1:n)', rbf_gram(X, X, opts.rbfGamma) + diag(1 ./ P.Ci)]; % RBF
        else
            Ktr = [(1:n)', full(X * X') + diag(1 ./ P.Ci)]; % linear
        end
    else
        Xsp = sparse(X);
    end

    out.skipped = false;

    % Starting point
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
        hist = rec_hist(hist, 0, plotted_objective(P, z0, ker.mul(z0), y));
    else
        hist = rec_hist(hist, 0, plotted_objective(P, z0, zeros(n, 1), y));
    end

    tolList = opts.smoTolerances(:)';
    a = zeros(n, 1);
    for t = 1:numel(tolList)
        args = sprintf('%s -e %.3g -q', base, tolList(t));
        tS = tic;
        if useKtil
            model = svmtrain(y, Ktr, args); %#ok<SVMTRAIN>
        else
            model = svmtrain(y, Xsp, args); %#ok<SVMTRAIN>
        end
        tTrain = toc(tS);

        a = zeros(n, 1);
        switch P.name
            case {'l1svm', 'l2svm'}
                a(model.sv_indices) = abs(model.sv_coef); % |y_i alpha_i| = alpha_i
            case 'nusvm'
                % LIBSVM rescales nu-SVC alphas internally; the direction
                % is exact, so restore the scale via the active constraint
                % sum(alpha) = nu, then repair the residual constraint
                % error exactly by projecting onto {0 <= a <= up,
                % per-class mass = nu/2} (two bisections, same structure
                % as Eq 23). Without the repair the small violations show
                % up as a fake accuracy floor for the baseline.
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
                a(model.sv_indices) = model.sv_coef; % beta = alpha - alpha*
        end

        f = plotted_objective(P, a, ker.mul(a), y);
        hist = rec_hist(hist, tTrain, f);
        maybe_print(opts, 'smo-libsvm', t, f);
        if tTrain >= opts.timeLimit
            break; % tighter tolerances only slower
        end
    end

    out.alpha = a;
    out.hist = hist;
end


function out = baseline_smo_mcsvm(ker, X, y, P, opts)
% Kernelized Crammer-Singer SMO baseline: maximal-violating-PAIR working set
% (two classes within one example) + analytic 2-variable step. This is the
% SMO *family* -- same as the LIBSVM binary baselines -- as opposed to the
% row-block coordinate descent in baseline_csdecomp_mcsvm (the LIBLINEAR -s 4
% family). Same dual, same ker operator => shares P* with solve_mcsvm.
    hist = init_hist();
    out  = struct('alpha', [], 'hist', hist, 'skipped', false);

    n = numel(y);   K = P.K;
    E   = full(sparse((1:n)', y, 1, n, K));
    CiK = repmat(P.Ci, 1, K);
    UP  = E .* CiK;            % [0, C_i] at true class
    LO  = (E - 1) .* CiK;      % [-C_i, 0] elsewhere

    if ker.explicit
        Kdiag = full(diag(ker.K));
    elseif strcmp(opts.kernel, 'rbf')
        Kdiag = ones(n, 1);            % exp(0) = 1
    else
        Kdiag = full(sum(X.^2, 2));    % linear
    end
    Kdiag = max(Kdiag, 1e-12);

    alpha = zeros(n, K);
    KA    = zeros(n, K);       % maintained K*alpha (= scores), alpha=0 -> 0
    epsB  = 1e-12;             % box-activity tolerance for eligibility
    clk   = clk_new(0);
    hist  = rec_hist(hist, 0, plotted_objective(P, alpha, KA, y));

    chkEvery = max(1, n);              % record / time-check cadence (~1 epoch)
    maxSteps = opts.maxIters * max(1, n);
    s = 0;
    while s < maxSteps
        s = s + 1;

        % ---- maximal violating pair, vectorized over all rows ----------
        G = KA - E;                                   % gradient K*alpha - E
        Gup = G;  Gup(alpha >= UP - epsB) = +inf;     % classes that can go UP
        Gdn = G;  Gdn(alpha <= LO + epsB) = -inf;     % classes that can go DOWN
        [GminUp, uIdx] = min(Gup, [], 2);
        [GmaxDn, vIdx] = max(Gdn, [], 2);
        viol = GmaxDn - GminUp;                       % KKT violation per row
        [mviol, i] = max(viol);
        if ~(mviol > opts.tol)                        % KKT-satisfied to tol
            break;
        end

        % ---- analytic 2-variable step in row i: a_u += t, a_v -= t ------
        u = uIdx(i);   v = vIdx(i);
        t = mviol / (2 * Kdiag(i));                                % free min
        t = min([t, UP(i,u) - alpha(i,u), alpha(i,v) - LO(i,v)]);  % clip box
        alpha(i,u) = alpha(i,u) + t;
        alpha(i,v) = alpha(i,v) - t;
        if ker.explicit
            Ki = ker.K(:, i);
        else
            Ki = ker.mul(sparse(i, 1, 1, n, 1));      % K(:,i) via one matvec
        end
        KA(:,u) = KA(:,u) + t * Ki;                   % refresh only 2 columns
        KA(:,v) = KA(:,v) - t * Ki;

        % ---- record / time gate ----------------------------------------
        if mod(s, chkEvery) == 0
            clk  = clk_pause(clk);
            f    = plotted_objective(P, alpha, KA, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, 'smo-cs', s, f);
            stop = clk.solve >= opts.timeLimit;
            clk  = clk_resume(clk);
            if stop, break; end
        end
    end

    clk  = clk_pause(clk);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, KA, y));
    out.alpha = alpha;
    out.hist  = hist;
end

% ===
% Helpers
% ===

function [theta, tk] = momentum_step(opts, mu, L, tk, restartStat)
% restartStat = <z - x_new, x_new - x_old>. > 0 means the momentum
% direction is fighting the latest projected step -> restart.
    theta = 0;
    if ~opts.accel
        return;
    end
    if mu > 0
        % Strongly convex: constant momentum from the known modulus,
        % beta = (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)). The gradient
        % restart is kept as a safeguard: it never hurts and helps when
        % the local curvature on the active manifold exceeds the global
        % mu.
        if restartStat > 0
            theta = 0;
        else
            rq = sqrt(mu / L);
            theta = (1 - rq) / (1 + rq);
        end
    else
        % mu = 0: adaptive restart-FIST A
        if restartStat > 0
            tk = 1;
        else
            tk1 = (1 + sqrt(1 + 4 * tk^2)) / 2;
            theta = (tk - 1) / tk1;
            tk = tk1;
        end
    end
end


function lam = bisect_root(h, scale)
    lo = -scale;
    hi = scale;
    k = 0;
    while h(lo) < 0 && k < 60
        hi = lo;
        lo = 2 * lo;
        k = k + 1;
    end
    k = 0;
    while h(hi) > 0 && k < 60
        lo = hi;
        hi = 2 * hi;
        k = k + 1;
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


% ======================================================================
% Solver clock (pauses during objective recording) and histories
% ======================================================================

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


% ======================================================================
% Figure: suboptimality (log scale) vs solver seconds. F* = best value
% seen by any method in this figure; the floor keeps log(0) at bay.
% ======================================================================

function make_config_figure(hists, labels, figPath, ttl, ylab)
    fstar = inf;
    for i = 1:numel(hists)
        if ~isempty(hists{i}.f)
            fstar = min(fstar, min(hists{i}.f));
        end
    end
    floorVal = max(1e-16, 1e-12 * max(1, abs(fstar)));

    fig = figure('Visible', 'off');
    hold on;
    styles = {'-', '-o', '-s', '-^'};
    for i = 1:numel(hists)
        H = hists{i};
        if isempty(H.f)
            continue;
        end
        semilogy(H.t, max(H.f - fstar, floorVal), styles{min(i, numel(styles))}, ...
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
