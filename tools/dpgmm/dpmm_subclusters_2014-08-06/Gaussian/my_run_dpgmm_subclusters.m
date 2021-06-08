function my_run_dpgmm_subclusters(data_file, model_dir, Mproc, sc)
% Modified by Thomas Schatz from:
% RUN_DPGMM_SUBCLUSTERS - runs the Dirichlet process Gaussian mixture model
% with subcluster splits and merges
%    run_dpgmm_subclusters(data, start, dispOn, Mproc, sc, as, alpha,
%    endtime, numits)
%
%    data - a DxN matrix containing double valued data, where D is the
%       dimension, and N is the number of data points
%    start - the number of initial clusters to start with
%    dispOn - whether progress should be continuously displayed
%    Mproc - the number of threads to use
%    sc - whether super-clusters are used or not
%    as - whether the approximate sampler is used or not
%    alpha - concentration parameter
%    endtime - the total time in seconds to stop after
%    numits - the number of iterations to stop after
%
%   Notes:
%     (1) The display shows the running time of the algorithm without I/O
%     to and from Matlab. It also doesn't show time to display. This is
%     more accurate, since one can always run as many iterations with C++
%     as desired.
%
%   [1] J. Chang and J. W. Fisher II, "Parallel Sampling of DP Mixtures
%       Models using Sub-Cluster Splits". Neural Information Processing
%       Systems (NIPS 2013), Lake Tahoe, NV, USA, Dec 2013.
%
%   Copyright(c) 2013. Jason Chang, CSAIL, MIT. 

addpath('../common');
addpath('include');

if (~exist('Mproc','var') || isempty(Mproc))
    Mproc = 1;
end
start=10;
if (~exist('sc','var') || isempty(sc))
    sc = true;
end
alpha=1;
as = false;
endtime = 10000000000;
numits = 1500;

raw_data = load(data_file);
fld = fieldnames(raw_data);
data = [];

for i = 1:length(fld)
    data = cat(1, data, raw_data.(fld{i}));
end

data = data';
N = size(data,2);
D = size(data,1);

if (D>N)
    error('More dimensions than observations.  Check data.');
end


params.alpha = alpha;
params.kappa = 1;
params.nu = D+3;
params.theta = mean(data,2);
params.delta = cov(data'); %eye(D);
params.its_crp = 20;
% params.its_ms = 1;
params.Mproc = Mproc;
params.useSuperclusters = logical(sc);
params.always_splittable = logical(as);

rng(0)
phi = rand(N,1)*start;
z = uint32(floor(phi));
disp('Initializing clusters');
clusters = initialize_clusters(data, phi, params);
disp('Clusters initialized');
numits = ceil(numits / params.its_crp);
time = zeros(numits*params.its_crp+1,1);
E = zeros(numits*params.its_crp+1,1);
E(1) = dpgmm_calc_posterior(data, z, params);
K = zeros(numits*params.its_crp+1,1);
K(1) = start;
NK = zeros(numits*params.its_crp+1,1);
NK(1) = max(hist(floor(phi), start));

colors = distinguishable_colors(50,[1 1 1]);

cindex = 1;
for it=1:numits
    if floor(log2(it))==log2(it)
        filename = fullfile(model_dir, int2str(cindex));
        save(filename, 'clusters', '-v7');
    end
    %disp('Phi(1:10): ');
    %disp(phi(1:10));
    [clusters, timediffs, Es, Ks, NKs] = dpgmm_subclusters(data, phi, clusters, params);
    %disp('Phi(1:10): ');
    %disp(phi(1:10));
    time(cindex+1:cindex+params.its_crp) = time(cindex) + cumsum(timediffs);
    E(cindex+1:cindex+params.its_crp) = Es;
    K(cindex+1:cindex+params.its_crp) = Ks;
    NK(cindex+1:cindex+params.its_crp) = NKs;
    cindex = cindex+params.its_crp;
        
    if (time(cindex)>endtime)
        break;
    end
    disp([num2str(cindex, '%04d') ' - ' num2str(mean(time(2:cindex)-time(1:cindex-1)),'%0.4f') ' - ' num2str(time(cindex))]);
end

%z = floor(phi);
%K = max(ceil(phi));
%logpi = {clusters(1:K).logpi};
%mu = {clusters(1:K).mu};
%Sigma = {clusters(1:K).Sigma};
%save(model_file, 'logpi', 'mu', 'Sigma', 'Es', '-v7');

filename = fullfile(model_dir, strcat(int2str(cindex), '-final'));
save(filename, 'clusters', '-v7');

