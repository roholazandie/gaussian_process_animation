function M = exp_map(mu, E)
% Computes exponential map on a sphere
%
% many thanks to Soren Hauberg!
D = size(E,1);
theta = sqrt(sum((E.^2)));
M = mu * cos(theta) + E .* repmat(sin(theta)./theta, D, 1);
if (any (abs (theta) <= 1e-7))
    for a = find (abs (theta) <= 1e-7)
        M (:, a) = mu;
    end % for
end % if
% M (:,abs (theta) <= 1e-7) = mu;
end % function