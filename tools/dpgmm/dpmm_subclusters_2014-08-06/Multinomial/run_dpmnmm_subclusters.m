function [z] = run_dpgmm_subclusters(data,start,dispOn,Mproc,sc,as,alpha, endtime, numits)
% RUN_DPMNMM_SUBCLUSTERS - runs the Dirichlet process multinomial mixture
% model with subcluster splits and merges
%    run_dpmnmm_subclusters(data, start, dispOn, Mproc, sc, as, alpha,
%    endtime, numits)
%
%    data - a DxN SPARSE matrix containing double valued data, where D is 
%       the dimension, and N is the number of data points
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

addpath('include');

if (~exist('dispOn','var') || isempty(dispOn))
    dispOn = false;
end
if (~exist('Mproc','var') || isempty(Mproc))
    Mproc = 1;
end
if (~exist('sc','var') || isempty(sc))
    sc = true;
end
if (~exist('as','var') || isempty(as))
    as = false;
end
if (~exist('endtime', 'var') || isempty(endtime))
    endtime = 1000;
end
if (~exist('numits', 'var') || isempty(numits))
    numits = 1000;
end

N = size(data,2);
D = size(data,1);

params.alpha = alpha;
params.diralpha = 1;
params.its_crp = 20;
% params.its_ms = 1;
params.Mproc = Mproc;
params.useSuperclusters = logical(sc);
params.always_splittable = logical(as);

phi = rand(N,1)*start;
z = uint32(floor(phi));
clusters = initialize_clusters(data, phi, params);

numits = 10000;
time = zeros(numits*params.its_crp+1,1);
E = zeros(numits*params.its_crp+1,1);
E(1) = dpmnmm_calc_posterior(data, z, params);
K = zeros(numits*params.its_crp+1,1);
K(1) = start;
NK = zeros(numits*params.its_crp+1,1);
NK(1) = max(hist(floor(phi), start));


cindex = 1;
for it=1:numits
    [clusters, timediffs, Es, Ks, NKs] = dpmnmm_subclusters(data, phi, clusters, params);
    time(cindex+1:cindex+params.its_crp) = time(cindex) + cumsum(timediffs);
    E(cindex+1:cindex+params.its_crp) = Es;
    K(cindex+1:cindex+params.its_crp) = Ks;
    NK(cindex+1:cindex+params.its_crp) = NKs;
    cindex = cindex+params.its_crp;
        
    if (time(cindex)>endtime)
        break;
    end
    
    if (dispOn)
        sfigure(1);
        subplot(2,1,1);
        plot(time(1:cindex),E(1:cindex));
        xlabel('Time (secs)');
        ylabel('Joint Log Likelihood');
        title(['Iteration: ' num2str(cindex) ' - Time: ' num2str(time(cindex))]);
        
        subplot(2,1,2);
        plot(floor(phi));
        title('Should look piecewise constant for synthetic');
        drawnow;
    end
    
    disp([num2str(cindex, '%04d') ' - ' num2str(mean(time(2:cindex)-time(1:cindex-1)),'%0.4f') ' - ' num2str(time(cindex))]);
end

z = floor(phi);