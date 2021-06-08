function [z] = run_dpgmm_fsd(data,start,dispOn,Mproc,alpha, endtime, numits)
% RUN_DPGMM_FSD - runs the Dirichlet process Gaussian mixture model
% with finite symmetric Dirichlet approximation
%    run_dpgmm_fsd(data, start, dispOn, Mproc, alpha, endtime, numits)
%
%    data - a DxN matrix containing double valued data, where D is the
%       dimension, and N is the number of data points
%    start - the number of initial clusters to start with
%    dispOn - whether progress should be continuously displayed
%    Mproc - the number of threads to use
%    alpha - concentration parameter
%    endtime - the total time in seconds to stop after
%    numits - the number of iterations to stop after
%
%   Notes:
%     (1) The display shows the running time of the algorithm without I/O
%     to and from Matlab. It also doesn't show time to display. This is
%     more accurate, since one can always run as many iterations with C++
%     as desired.
%     (2) Change params.K for the truncation approximation.
%
%   [1] J. Chang and J. W. Fisher II, "Parallel Sampling of DP Mixtures
%       Models using Sub-Cluster Splits". Neural Information Processing
%       Systems (NIPS 2013), Lake Tahoe, NV, USA, Dec 2013.
%   [2] H. Ishwaran and M. Zarepour. Exact and Approximate
%       Sum-Representations for the Dirichlet Process. Canadian Journal of
%       Statistics, 30:269-283, 2002.
%
%   Copyright(c) 2013. Jason Chang, CSAIL, MIT. 

addpath('include');

if (~exist('dispOn','var') || isempty(dispOn))
    dispOn = false;
end
if (~exist('Mproc','var') || isempty(Mproc))
    Mproc = 1;
end
if (~exist('endtime', 'var') || isempty(endtime))
    endtime = 1000;
end
if (~exist('numits', 'var') || isempty(numits))
    numits = 1000;
end

N = size(data,2);
D = size(data,1);

if (D>N)
    error('More dimensions than observations.  Check data.');
end


% params.alpha = 1e-3;10
params.alpha = alpha;
params.kappa = 1;
params.nu = D+3;
params.theta = mean(data,2);
params.delta = eye(D);
params.its_crp = 50;
params.its_ms = 1;
params.Mproc = Mproc;
params.K = 100;

z = uint32(randi(start,[N,1])-1);

time = zeros(numits+1,1);
E = zeros(numits+1,1);
E(1) = dpgmm_calc_posterior(data, z, params);

colors = distinguishable_colors(params.K,[1 1 1]);

cindex = 1;
for it=1:numits
    [timediffs, Es] = dpgmm_FSD(data, z, params);
    time(cindex+1:cindex+params.its_crp) = time(cindex) + cumsum(timediffs);
    E(cindex+1:cindex+params.its_crp) = Es;
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
        hold off;
        for k=1:params.K
            mask = z==k-1;
            plot(data(1,mask), data(2,mask), 'o', 'Color', colors(k,:));
            hold on;
        end
        drawnow;
    end
    
    
    disp([num2str(cindex, '%04d') ' - ' num2str(mean(time(2:cindex)-time(1:cindex-1)),'%0.4f') ' - ' num2str(time(cindex))]);
end