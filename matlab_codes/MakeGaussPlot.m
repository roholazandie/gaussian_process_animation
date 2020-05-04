function MakeGaussPlot(kxx,filename,phi)
% the standard gauss plot, using the nonlinear dataset
% Philipp Hennig, 11 Dec 2012
dgr = [0,0.4717,0.4604]; % color [0,125,122]
dre = [0.4906,0,0]; % color [130,0,0]
lightdgr = [1,1,1] - 0.5 * ([1,1,1] - dgr);
lightdre = [1,1,1] - 0.5 * ([1,1,1] - dre);
dgr2white = bsxfun(@minus,[1,1,1],bsxfun(@times,(linspace(0,0.6,2024)').^0.5,[1,1,1]-dgr));
dre2white = bsxfun(@minus,[1,1,1],bsxfun(@times,(linspace(0,0.6,2024)').^0.5,[1,1,1]-dre));
%addpath ˜/Documents/MATLAB/matlab2tikz-0.1.2/
M = 150; % # dimensions


F = 100; % # frames
x = linspace(-8,8,M)';
s1 = GPanimation(M,F);
s2 = GPanimation(M,F);
s3 = GPanimation(M,F);
GaussDensity = @(y,m,v)(bsxfun(@rdivide,exp(-0.5*bsxfun(@rdivide,bsxfun(@minus,y,m').^2,v'))./sqrt(2*pi),sqrt(v')));
%% prior
%kxx = k(x,x); % kernel function (enter your favorite here)
m = zeros(M,1);
V = kxx;
L = chol(V + 1.0e-8 * eye(M)); % jitter for numerical stability
y = linspace(-15,20,250)';
P = GaussDensity(y,m,diag(V+eps));
%P = mvnpdf(y', m', diag(V+eps));
colormap(dgr2white);
for f = 1:F
    clf;
    hold on
    imagesc(x,y,P);
    plot(x,max(min(m,20),-15),'-','Color',dgr,'LineWidth',0.7);
    plot(x,max(min(m + 2 * sqrt(diag(V)),20),-15),'-','Color',lightdgr,'LineWidth',.5);
    plot(x,max(min(m - 2 * sqrt(diag(V)),20),-15),'-','Color',lightdgr,'LineWidth',.5);
    if nargin > 2
        plot(x,phi(x),'-','Color',0.7*ones(3,1));
    end
    plot(x,m + L' * s1(:,f),'--','Color',dre);
    plot(x,m + L' * s2(:,f),'--','Color',dre);
    plot(x,m + L' * s3(:,f),'--','Color',dre);
    xlim([-8,8]);
    ylim([-15,20]);
    drawnow; 
    pause(0.01)
end