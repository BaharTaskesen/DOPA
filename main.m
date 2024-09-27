clc 
clear all
close all

reps = 10;
K_max = 1e4;

ind = (1:9)' * logspace(0, log10(K_max)-1, log10(K_max)); 
ind = [ind(:)', K_max];

[~, K_size] = size(ind);

ftrl_timer = zeros(reps, K_size);
bisection_timer = zeros(reps, K_size);

rng("default")

prm = {};
prm.eta = 1;
prm.method = "tsallis";
prm.eps_ = 1e-8;

save('ind')

for rep = 1 : reps
    rep
    for K_ind = 1 : K_size
        K_ind
        K = ind(K_ind);
        prm.K = K;
        u_hat = rand(K, 1); 
        bsc = @() bisection(u_hat', prm);
        bisection_timer(rep, K_ind) = timeit(bsc);
        ftrl_fcn = @() ftrl(u_hat, prm);
        ftrl_timer(rep, K_ind) = timeit(ftrl_fcn);
    end
    save('ftrl_timer')
    save('bisection_timer')
end

save('ftrl_timer')
save('bisection_timer')

