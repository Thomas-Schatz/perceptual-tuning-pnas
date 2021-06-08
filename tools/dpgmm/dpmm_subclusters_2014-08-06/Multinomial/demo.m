addpath('../common');

% set the random number generation seed for reproducible data
RandStream.setGlobalStream(RandStream('mt19937ar','Seed', 1));

% generate data
data = generate_mn_data();

% run the sampler
initialClusters = 1;
dispOn = true;
numProcessors = 8;
useSuperclusters = false;
approximateSampling = false;
alpha = 1;
endtime = 1000;
numits = 10000;

% uncomment the algorithm you want to run

run_dpmnmm_subclusters(data, initialClusters, dispOn, numProcessors, ...
    useSuperclusters, approximateSampling, alpha, endtime, numits);

% run_dpmnmm_fsd(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);

% run_dpmnmm_sams(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);

% run_dpmnmm_gibbs(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);

% there seem to be some convergence issues with the supercluster method
% this may be an implementation issue since random orderings aren't used
% we haven't seen the issue in the real datasets that were tested
% run_dpmnmm_gibbs_superclusters(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);