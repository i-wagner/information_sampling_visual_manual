function Pret = colnoise(N, alpha)
%COLNOISE  Generate colored noise (noise with 1/f^alpha amplitude spectrum)
%
%    alpha = 1: pink noise (default)
%            2: brown noise
%            0: white noise
%            -1: blue  
%            -2: purple
%  
%Example 1: Generate pink noise image
%  P = colnoise(512);
%
%Example 2: plot statistics (N=128, alpha =2)
%  colnoise;
% 
%Example 3: Analys and manipulate frequencies in an image
%   I = double(imread('lena.tif'))/255; 
%   colnoise(I, -1);
%  
% (c) 2004-09-22  Thorsten Hansen

if nargin < 1
  I = randn(128);
elseif prod(size(N)) > 1 % not a single value, interpret as image
  I = N;
else
  I = rand(N);
end

if nargin < 2
  alpha = 1;
end

Ny = size(I, 1);
Nx = size(I, 2);


% wave numbers
kx = [0:Nx/2 floor(-Nx/2)+1:-1];
ky = [0:Ny/2 floor(-Ny/2)+1:-1];
[Kx, Ky] = meshgrid(kx, ky);
% frequencies
F = sqrt(Kx.^2 + Ky.^2);

% colored noise
F1 = F; F1(1, 1) = 1;
P = real(ifft2(fft2(I)./F1.^alpha));

% scale P to [0;1]
P = P - min(min(P));
P = P / max(max(P));


%
% plot statistics (if called w/o output arguments)
%
if nargout == 0
  subplot(2,2,1), imshow(I, []), title('random noise')
  subplot(2,2,3), imshow(P, []), title('colored noise')
  drawnow
  if N>128
    disp('... wait for plotting frequency distribution')
  end
  subplot(2,2,2), plot_freq_dist(I, F, 0); drawnow
  subplot(2,2,4), plot_freq_dist(P, F, alpha); drawnow
end

if nargout > 0
  Pret = P;
end

%------------------------------------------------------------------------------
function plot_freq_dist(I, F, alpha)
%------------------------------------------------------------------------------
  
% available frequencies
f = sort(unique(F(:))); 

% for each frequency: determine energy from amplitude spectrum
IA = abs(fft2(I));
af = zeros(1, length(f));
for i=1:length(f)
  af(i) = mean(IA(find(F==f(i))));
end
  
%loglog(f, af, '.')
%xlabel('frequency')
%ylabel('mean energy')
%hold on
%loglog(f(2:end), af(2)./(f(2:end).^alpha), 'r') 
%hold off
%axis tight

% regression 

x = log(f(2:end)+eps);
y = log(af(2:end)+eps)';

scatter(x,y, '.')
hold on


% give not a nice fit...
%bls = regress(x, [ones(length(y),1) y])
%plot(x, bls(1)+bls(2)*x, 'r:', 'Linewidth', 4);

brob = robustfit(x,y);
plot(x, brob(1)+brob(2)*x, 'r-', 'Linewidth', 1);
xlabel('log frequency')
ylabel('log mean energy')
axis tight

disp(num2str(brob(2)))

hold off
