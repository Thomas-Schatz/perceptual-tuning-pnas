function [data] = generate_mn_data()

N = 1e4;
D = 100;

K = randi(5)+2;
K = 3;
data = zeros(N,D);
for k=1:K
    pi = dirrnd(ones(D,1)*0.05);
    
    istart = floor(N/K*(k-1)) + 1;
    istop = floor(N/K*k);
    
    numWords = randi(1000,istop-istart+1,1);
    data(istart:istop,:) = mnrnd(numWords,pi);
end

data = data';
data = sparse(data);