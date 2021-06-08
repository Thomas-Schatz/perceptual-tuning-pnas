function [clusters] = initialize_clusters(data, phi, params)

K = max(ceil(phi));

clusters = struct('logpi', [], ...
    'logpi_mn',[],'logpi_mn_l',[],'logpi_mn_r',[],...
    'logsublikelihood',[],'logsublikelihoodDelta',[],'splittable',[]);

Nk = zeros(K+1,1);
D = size(data,1);
for k=1:K
    indices = ceil(phi)==k;
    Nk(k) = nnz(indices);
    indices_l = indices & (phi-k+1<0.5);
    indices_r = indices & ~indices_l;
    
    clusters(k).logpi_mn = ones(D,1)/D;
    clusters(k).logpi_mn_l = ones(D,1)/D;
    clusters(k).logpi_mn_r = ones(D,1)/D;
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