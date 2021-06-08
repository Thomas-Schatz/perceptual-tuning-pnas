function [clusters] = initialize_clusters(data, phi, params)

K = max(ceil(phi));

clusters = struct('z', [], 'logpi', [], ...
    'mu',[],'mu_l',[],'mu_r',[],...
    'Sigma',[],'Sigma_l',[],'Sigma_r',[],...
    'logsublikelihood',[],'logsublikelihoodDelta',[],'splittable',[]);

Nk = zeros(K+1,1);
for k=1:K
    indices = ceil(phi)==k;
    Nk(k) = nnz(indices);
    indices_ll = indices & (phi-k+1<0.25);
    indices_lr = indices & ~indices_ll & (phi-k+1<0.5);
    indices_rl = indices & ~indices_ll & ~indices_lr & (phi-k+1<0.75);
    indices_rr = indices & ~indices_ll & ~indices_lr & ~indices_rl;
    
    indices_l = indices_ll | indices_lr;
    indices_r = indices_rl | indices_rr;
    
    clusters(k).z = k-1;
    clusters(k).mu = mean(data(:,indices),2);
    clusters(k).Sigma = cov(data(:,indices)');
    clusters(k).mu_l = mean(data(:,indices_l),2);
    clusters(k).Sigma_l = cov(data(:,indices_l)');
    clusters(k).mu_r = mean(data(:,indices_r),2);
    clusters(k).Sigma_r = cov(data(:,indices_r)');
    clusters(k).logsublikelihood = -inf;
    clusters(k).logsublikelihoodDelta = inf;
    clusters(k).splittable = false;
end
Nk(end) = params.alpha;

[logpi] = dirrnd(Nk);
logpi = log(logpi);
for k=1:K
    clusters(k).logpi = logpi(k);
end