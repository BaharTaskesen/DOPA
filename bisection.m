function [p_bisect, delta_eps] = bisection(u, param)
    eps_ = param.eps_;
    N = param.K;
    eta = param.eta;
    if param.method == "entropic"
        F = @(s) exp(s - 1);
        F_i_s = @(s) min(ones(size(s)), max(zeros(size(s)), 1 - F(-s./eta)));
        temp = - eta.*log(1/N) - 1;
        L = eta * exp(-1);
    else 
        q = 1/2;
        F = @(s) (s * (q-1) / q + 1/q) .^ (1 / (q-1));
        F_i_s = @(s) min(ones(size(s)), max(zeros(size(s)), 1 - F(-s / eta)));
        temp = - (2 - N ^ (1/2)) * param.eta;
        L =  2 ^ (-2);
    end
        
    tau_u = max(u - temp, [], 2);
    tau_l = min(u - temp, [], 2);
    delta_eps = eps_ / L / sqrt(N);
   
    for k = 1 : ceil(log2(max((tau_u - tau_l) / delta_eps))) 
        tau = (tau_u + tau_l) ./ 2;
        p = 1 - F_i_s(u - tau);
        indx = find(sum(p, 2) > 1);
        tau_u(indx) = tau(indx);
        indx2 = find(sum(p, 2) <= 1);
        tau_l(indx2) = tau(indx2);
    end

    p_bisect = 1 -  F_i_s(u - tau_l);
    p_bisect = p_bisect';
end