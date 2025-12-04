function p = ftrl(u, prm)
    [s1 , ~] = size(u);
    K = s1;
    p = sdpvar(K, 1);
    z = sdpvar(K, 1);

    % Define constraints
    cons = [sum(p) == 1, p >= 0, p <= 1];

    % Add second-order cone constraints
    for i = 1:K
        cons = [cons, cone([2*z(i); 1 - p(i)], 1 + p(i))];
    end

    obj = sum(p .* u) + prm.eta * 2 * sum(z - p);


    ops = sdpsettings('verbose', 0, 'solver', 'mosek');
    optimize (cons, -obj, ops);
end