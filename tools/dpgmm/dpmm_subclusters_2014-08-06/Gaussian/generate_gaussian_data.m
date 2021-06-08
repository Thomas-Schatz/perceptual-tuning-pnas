function [data] = generate_gaussian_data()

N = 10000; % number of data points
D = 2; % number of dimensions
K = 5+randi(1); % number of components

x = randn(D,N);

tpi = dirrnd(ones(K,1));
tzn = mnrnd(N, tpi);
tz = zeros(N,1);

tmean = zeros(D,K);
tcov = zeros(D,D,K);

ind = 1;
for i=1:numel(tzn)
    indices = ind:ind+tzn(i)-1;
    tz(indices) = i;
    
    tmean(:,i) = mvnrnd(zeros(D,1), 100*eye(D));
    tcov(:,:,i) = iwishrnd(eye(D)*1,D+2);
    T = cholcov(tcov(:,:,i));
    
    x(:,indices) = bsxfun(@plus, T*x(:,indices), tmean(:,i));
    
    ind = ind+tzn(i);
end

% if (D==1)
%     scatter(x,x);
% else
%     scatter(x(1,:), x(2,:));
% end
% drawnow;
data = x;
