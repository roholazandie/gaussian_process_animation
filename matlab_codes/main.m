clc;
clear;

M = 150; % # dimensions
F = 20; % # frames
x = linspace(-8,8,M)';
% s1 = GPanimation(M,F);
% 
% for i=1:20
%     plot(x, s1(:, i))
%     pause(0.2)
% end

se_length_scale = 2.5;
se_outout_var = 2;
se_kernel = @(x,y) se_outout_var*exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ se_length_scale^2 );

K = bsxfun(se_kernel, x', x );
MakeGaussPlot(K)