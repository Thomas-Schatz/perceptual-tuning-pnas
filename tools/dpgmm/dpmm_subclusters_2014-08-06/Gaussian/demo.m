addpath('../common');

% set the random number generation seed for reproducible data
RandStream.setGlobalStream(RandStream('mt19937ar','Seed', 8));

% generate data
data = generate_gaussian_data();

% run the sampler
initialClusters = 10;
dispOn = false;
numProcessors = 12;
useSuperclusters = false;
approximateSampling = false;
alpha = 1;
endtime = 10000;
numits = 10000;

% uncomment the algorithm you want to run

run_dpgmm_subclusters(data, initialClusters, dispOn, numProcessors, ...
    useSuperclusters, approximateSampling, alpha, endtime, numits);

% run_dpgmm_fsd(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);

% run_dpgmm_sams(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);

% run_dpgmm_gibbs(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);

% run_dpgmm_gibbs_superclusters(data, initialClusters, dispOn, numProcessors, ...
%     alpha, endtime, numits);
