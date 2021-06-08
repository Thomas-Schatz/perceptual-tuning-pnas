function [] = run_dpgmm_gibbs_superclusters(data,start,dispOn,Mproc,alpha, endtime, numits)
% RUN_DPGMM_GIBBS_SUPERCLUSTERS - runs the Dirichlet process Gaussian 
% mixture model while marginalizing pi and theta and using superclusters
%    run_dpgmm_gibbs_superclusters(data, start, dispOn, Mproc, alpha,
%       endtime, numits)
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
%
%   [1] J. Chang and J. W. Fisher II, "Parallel Sampling of DP Mixtures
%       Models using Sub-Cluster Splits". Neural Information Processing
%       Systems (NIPS 2013), Lake Tahoe, NV, USA, Dec 2013.
%   [2] D. Lovell, J. Malmaud, R. P. Adams, and V. K. Mansinghka. Parallel
%       Markov Chain Monte Carlo for Dirichlet Process Mixtures. Workshop
%        on Big Learning, NIPS, 2012.
%   [3] S. A. Williamson, A. Dubey, and E. P. Xing. Parallel Markov
%       Chain Monte Carlo for Nonparametric Mixture Models. International
%       Conference on Machine Learning, 2013.
%
%   Copyright(c) 2013. Jason Chang, CSAIL, MIT. 

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
params.its_crp = 1;
params.its_ms = 0;
params.Mproc = Mproc;

phi = rand(N,1)*start;
z = uint32(floor(phi));

numits = 10000;
time = zeros(numits+1,1);
E = zeros(numits+1,1);
E(1) = dpgmm_calc_posterior(data, z, params);
sc = int32(0:numel(phi)-1);

colors = distinguishable_colors(50,[1 1 1]);

for it=1:numits
    time(it+1) = time(it) + dpgmm_sams_superclusters(data, z, sc, params);
    E(it+1) = dpgmm_calc_posterior(data, z, params);
    
    if (time(it+1)>endtime)
        break;
    end
    
    if (dispOn)
        if (max(z(:))+1 > size(colors,1))
            c = max(max(z(:))+1, 2*size(colors,1));
            colors = distinguishable_colors(c,[1 1 1]);
        end
        
        sfigure(1);
        subplot(2,1,1);
        plot(time(1:it+1),E(1:it+1));
        xlabel('Time (secs)');
        ylabel('Joint Log Likelihood');
        title(['Iteration: ' num2str(it) ' - Time: ' num2str(time(it+1))]);

        subplot(2,1,2);
        hold off;
        for k=0:max(z(:))
            mask = z==k;
            plot(data(1,mask), data(2,mask), 'o', 'Color', colors(k+1,:));
            hold on;
        end
        drawnow;
    end
    
    disp([num2str(it, '%04d') ' - ' num2str(mean(time(2:it+1)-time(1:it)),'%0.4f') ' - ' num2str(time(it+1))]);
end

E = E(1:it+1);
time = time(1:it+1);
save(name,'E','time','z');
% save(['output_NIPS/sams_' num2str(start,'%03d') '_' num2str(sample_it,'%03d') '.mat'],'E','time','phi');
