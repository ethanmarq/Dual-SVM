function results = run_svm_comparison(matFile, opts)
    addpath('./libsvm-336/matlab');
    addpath('./liblinear-249/matlab');
%RUN_SVM_COMPARISON Dual projected-gradient SVM solvers vs SMO baselines.
% run_svm_comparison('dataset.mat');
% output: <outRoot>/<dataset>/<problem>_<costMode>.png
%
% Paper's Proposed methods:
%   'l1svm'  Eq 5 dual, box [0, C_i],   step  beta = a - (K a - 1)/L,
%            L = sigma_1(K); projection: find lambda with
%            sum_i y_i min{C_i, max{beta_i - lambda y_i, 0}} = 0 (bisection)
%
%   'l2svm'  Eq 5 dual with + 1/(2C_i) ||a||^2 term, a >= 0 (no upper box),
%            step beta = a - (K a + a./C_i - 1)/L, L = sigma_1(K) + 1/min(C_i);
%            projection: sum_i y_i [beta_i - lambda y_i]_+ = 0
%
%   'svr'    Eq 10 dual: 0.5 b'Kb - y'b + eps ||b||_1, 1'b = 0, box;
%            v = b - (K b - y)/L, then b = clip(S_{eps/L}(v - lambda), lo, up)
%            with lambda from sum_i clip(...) = 0 (bisection)
%
%   'mcsvm'  Eq 16 Jacobi-style parallel update: B = a - (K a - 1_ind)/L,
%            then per-row bisection for lambda_i (vectorized across rows)
%
%   'nusvm'  Eq 21-23: B = a - (K a)/L, project onto {y'a = 0, 0<=a<=up}
%            (Eq 21); if sum(a) < nu, enforce the per-class mass
%            constraints sum_{y=+1} a = sum_{y=-1} a = nu/2 (Eq 23) via two
%            bisections and update by Eq 22
%
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
% Cost-sensitive mode (opts.costMode = 'cost')
%   classification : per-class box C_i = C * w_{y_i}; w defaults to
%                    balanced weights n/(K n_k), override w/ opts.classCosts
%   nusvm          : per-class box  up_i = w_{y_i}/n  (Eq 22/23 kept as-is)
%   svr            : asymmetric box [-C*costNegFactor, +C*costPosFactor]
%                    (different penalty for over-/under-estimation)
%
% Kernel handling: linear kernel throughout. K is formed explicitly when
% n <= opts.explicitKernelMaxN, otherwise K*alpha is applied implicitly
% as y.*(X*(X'*(y.*alpha))) (classification) or X*(X'*alpha) (svr/mcsvm),
% which is what makes mnist8m-sized runs possible. sigma_1(K) =
% sigma_max(X)^2 via svds.
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
    P.chargedSetup = ker.setupTime;   % sigma_1 / K build time billed to us
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
    propLabel = sprintf('Proposed PG dual (%s) [RENAME]', P.name);

    % ---- baseline --------------------------------------------------------
    if strcmp(P.name, 'mcsvm')
        baseOut = baseline_liblinear_mcsvm(X, y, P, opts);
        baseLabel = 'LIBLINEAR -s 4 (CS dual CD, tol sweep)';
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
    if strcmp(P.name, 'mcsvm')
        ylab = 'Primal suboptimality  P - P*';
    else
        ylab = 'Dual suboptimality  f - f*';
    end
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
    opts = set_default(opts, 'problem', 'l1svm');
    opts = set_default(opts, 'costMode', 'none');
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
    opts = set_default(opts, 'evalEvery', 1);
    opts = set_default(opts, 'printEvery', 200);
    opts = set_default(opts, 'maxSamples', inf);
    opts = set_default(opts, 'standardize', true);
    opts = set_default(opts, 'standardizeY', true);
    opts = set_default(opts, 'explicitKernelMaxN', 8000);
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
% Sniff (X, y) from common variable names, fall back to shape matching.
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
% Kernel operator: explicit K for small n, implicit linear matvec else.
% Classification uses the label-signed kernel K = diag(y) X X' diag(y);
% svr / mcsvm use the plain Gram K = X X'. sigma_1 is identical for both.
% ======================================================================

function ker = make_kernel_op(X, y, P, opts)
    n = size(X, 1);
    signed = any(strcmp(P.name, {'l1svm', 'l2svm', 'nusvm'}));
    tSetup = tic;

    if n <= opts.explicitKernelMaxN
        K = full(X * X');
        if signed
            K = (y * y') .* K;
        end
        ker.mul = @(a) K * a;
        ker.explicit = true;
        ker.K = K;
    else
        Xt = X'; % cache the transpose once
        if signed
            ker.mul = @(a) y .* (X * (Xt * (y .* a)));
        else
            ker.mul = @(a) X * (Xt * a); % works for n x K matrices too
        end
        ker.explicit = false;
    end

    % sigma_1(K) = sigma_max(X)^2 (diag(y) is orthogonal, so the signed
    % kernel has the same spectrum as X X').
    try
        ker.sig1 = svds(X, 1)^2;
    catch
        v = randn(n, 1);
        for k = 1:50
            v = ker.mul(v);
            v = v / max(norm(v), 1e-300);
        end
        ker.sig1 = 1.05 * (v' * ker.mul(v)); % safety factor: power
    end % iteration underestimates
    ker.sig1 = max(ker.sig1, 1e-12);
    ker.setupTime = toc(tSetup);
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
            % PRIMAL of the Crammer-Singer problem (see header): with
            % W = X'a we have ||W||_F^2 = <a, g> and scores S = XW = g.
            n = numel(y);
            idxTrue = sub2ind(size(g), (1:n)', y);
            Drow = 1 - full(sparse((1:n)', y, 1, n, P.K));   % Delta(y_i,:)
            loss = max(Drow + g, [], 2) - g(idxTrue);
            f = 0.5 * sum(sum(a .* g)) + sum(P.Ci .* loss);
    end
end


% ======================================================================
% Solvers
% ======================================================================

function out = solve_l1l2(ker, y, P, opts)
    n = numel(y);
    isL2 = strcmp(P.name, 'l2svm');
    if isL2
        L = ker.sig1 + 1 / min(P.Ci);
    else
        L = ker.sig1;
    end

    alpha = zeros(n, 1);
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        g = ker.mul(alpha); % on the clock
        if isL2
            grad = g + alpha ./ P.Ci - 1;
        else
            grad = g - 1;
        end

        clk = clk_pause(clk); % evaluation off-clock
        if mod(it - 1, opts.evalEvery) == 0
            f = plotted_objective(P, alpha, g, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        beta = alpha - grad / L;
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

        converged = norm(alphaNew - alpha) <= opts.tol * max(1, norm(alpha));
        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    g = ker.mul(alpha); % final point, off-clock
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, g, y));

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end


function out = solve_svr(ker, y, P, opts)
    n = numel(y);
    L = ker.sig1;
    thr = P.eps / L;

    b = zeros(n, 1);
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        g = ker.mul(b);
        grad = g - y;

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f = plotted_objective(P, b, g, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        v = b - grad / L;
        st = @(u) sign(u) .* max(abs(u) - thr, 0);   % soft threshold S_{eps/L}
        h = @(lam) sum(min(P.bUp, max(P.bLo, st(v - lam))));
        lam = bisect_root(h, max(1, max(abs(v))));
        bNew = min(P.bUp, max(P.bLo, st(v - lam)));

        converged = norm(bNew - b) <= opts.tol * max(1, norm(b));
        b = bNew;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    g = ker.mul(b);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, b, g, y));

    out = struct('alpha', b, 'hist', hist, 'iters', it, 'skipped', false);
end


function out = solve_mcsvm(ker, y, P, opts)
    n = numel(y);
    K = P.K;
    L = ker.sig1;

    E = full(sparse((1:n)', y, 1, n, K));            % 1_ind
    CiK = repmat(P.Ci, 1, K);
    UP = E .* CiK;                                   % [0, C_i] at true class
    LO = (E - 1) .* CiK;                             % [-C_i, 0] elsewhere

    alpha = zeros(n, K);
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        G = ker.mul(alpha);                          % n x K

        clk = clk_pause(clk);
        if mod(it - 1, opts.evalEvery) == 0
            f = plotted_objective(P, alpha, G, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        B = alpha - (G - E) / L;

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

        converged = norm(alphaNew(:) - alpha(:)) <= opts.tol * max(1, norm(alpha(:)));
        alpha = alphaNew;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    G = ker.mul(alpha);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, G, y));

    out = struct('alpha', alpha, 'hist', hist, 'iters', it, 'skipped', false);
end


function out = solve_nusvm(ker, y, P, opts)
    n = numel(y);
    L = ker.sig1;
    up = P.up;
    ip = (y > 0);
    im = ~ip;

    alpha = zeros(n, 1);
    hist = init_hist();
    clk = clk_new(P.chargedSetup);

    it = 0;
    while it < opts.maxIters
        it = it + 1;

        g = ker.mul(alpha);

        clk = clk_pause(clk);
        % alpha = 0 (it == 1) violates sum(alpha) >= nu and has f = 0 < f*,
        % which would wreck a log suboptimality plot, so recording starts
        % once the iterate is feasible (after the first projection).
        if it > 1 && mod(it - 1, opts.evalEvery) == 0
            f = plotted_objective(P, alpha, g, y);
            hist = rec_hist(hist, clk.solve, f);
            maybe_print(opts, P.name, it, f);
        end
        stop = clk.solve >= opts.timeLimit;
        clk = clk_resume(clk);
        if stop
            break;
        end

        B = alpha - g / L;

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

        converged = norm(a - alpha) <= opts.tol * max(1, norm(alpha));
        alpha = a;
        if converged
            break;
        end
    end

    clk = clk_pause(clk);
    g = ker.mul(alpha);
    hist = rec_hist(hist, clk.solve, plotted_objective(P, alpha, g, y));

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

    switch P.name
        case 'l1svm'
            base = sprintf('-s 0 -t 0 -c %.10g', P.C);
            if strcmp(P.costMode, 'cost')
                % per-class boxes via -wi (box becomes C * w_class)
                wp = P.Ci(find(y > 0, 1)) / P.C;
                wm = P.Ci(find(y < 0, 1)) / P.C;
                base = sprintf('%s -w1 %.10g -w-1 %.10g', base, wp, wm);
            end
        case 'l2svm'
            % L2-SVM == hard-margin SMO on K + diag(1./C_i): solve with a
            % precomputed kernel and an effectively infinite box.
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
            base = sprintf('-s 1 -t 0 -n %.10g', P.nu);
        case 'svr'
            if strcmp(P.costMode, 'cost')
                warning('Stock LIBSVM eps-SVR has one C (no asymmetric box); skipping baseline.');
                return;
            end
            base = sprintf('-s 3 -t 0 -c %.10g -p %.10g', P.C, P.eps);
        otherwise
            return;
    end

    % One-off conversions, excluded from the recorded training times.
    if useKtil
        Ktr = [(1:n)', full(X * X') + diag(1 ./ P.Ci)];
    else
        Xsp = sparse(X);
    end

    out.skipped = false;

    % Starting point of any iterative method: alpha = 0 (skip for nusvm,
    % where alpha = 0 is infeasible and f(0) = 0 < f*).
    if ~strcmp(P.name, 'nusvm')
        z0 = zeros(n, 1);
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
                a(model.sv_indices) = abs(model.sv_coef);   % |y_i alpha_i| = alpha_i
            case 'nusvm'
                % LIBSVM rescales nu-SVC alphas internally (by 1/rho'); the
                % direction is exact, so restore the scale via the active
                % constraint sum(alpha) = nu.   TODO(verify) once vs a
                % small run: alpha should satisfy 0 <= alpha <= 1/n.
                a(model.sv_indices) = abs(model.sv_coef);
                s = sum(a);
                if s > 0
                    a = a * (P.nu / s);
                end
                a = min(a, P.up);
            case 'svr'
                a(model.sv_indices) = model.sv_coef;        % beta = alpha - alpha*
        end

        f = plotted_objective(P, a, ker.mul(a), y);
        hist = rec_hist(hist, tTrain, f);
        maybe_print(opts, 'smo-libsvm', t, f);
        if tTrain >= opts.timeLimit
            break;                          % tighter tolerances only slower
        end
    end

    out.alpha = a;
    out.hist = hist;
end


function out = baseline_liblinear_mcsvm(X, y, P, opts)
% Crammer-Singer baseline via LIBLINEAR -s 4 (sequential dual coordinate
% descent). LIBLINEAR never exposes its dual variables, so this baseline
% is scored on the shared PRIMAL objective from the returned W (their
% -s 4 problem is bias-free, exactly like Eq 16).
    hist = init_hist();
    out = struct('W', [], 'hist', hist, 'skipped', true);

    fn = opts.liblinearFn;
    if exist(fn, 'file') ~= 3
        warning('LIBLINEAR mex (%s) not on path; skipping CS baseline.', fn);
        return;
    end

    n = size(X, 1);
    d = size(X, 2);
    base = sprintf('-s 4 -c %.10g', P.C);
    if strcmp(P.costMode, 'cost')
        for k = 1:P.K
            wk = P.Ci(find(y == k, 1)) / P.C;
            base = sprintf('%s -w%d %.10g', base, k, wk);
        end
    end

    Xsp = sparse(X);
    idxTrue = sub2ind([n, P.K], (1:n)', y);
    Drow = 1 - full(sparse((1:n)', y, 1, n, P.K));
    out.skipped = false;

    % alpha = 0 <=> W = 0 starting point
    S0 = zeros(n, P.K);
    f0 = sum(P.Ci .* (max(Drow + S0, [], 2) - S0(idxTrue)));
    hist = rec_hist(hist, 0, f0);

    W = zeros(d, P.K);
    tolList = opts.smoTolerances(:)';
    for t = 1:numel(tolList)
        args = sprintf('%s -e %.3g -q', base, tolList(t));
        tS = tic;
        model = feval(fn, y, Xsp, args);
        tTrain = toc(tS);

        % TODO(verify): w shape / Label order differ across liblinear
        % versions; check once on a tiny 3-class example.
        Wm = model.w;
        if size(Wm, 1) == P.K && size(Wm, 2) == d
            Wm = Wm';
        end
        W = zeros(d, P.K);
        W(:, model.Label) = Wm;

        S = X * W;
        f = 0.5 * sum(W(:).^2) + sum(P.Ci .* (max(Drow + S, [], 2) - S(idxTrue)));
        hist = rec_hist(hist, tTrain, f);
        maybe_print(opts, 'liblinear-cs', t, f);
        if tTrain >= opts.timeLimit
            break;
        end
    end

    out.W = W;
    out.hist = hist;
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
    close(fig);
end
