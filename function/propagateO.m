function [w_o] = propagateO(w_i, dist, pxsize, wavlen)

% dimension of the wavefield
[ny,nx] = size(w_i);

% sampling in the frequency domain
kx = pi/pxsize*(-1:2/nx:1-2/nx);%³¤¶ÈÎªN1
ky = pi/pxsize*(-1:2/ny:1-2/ny);
[KX,KY] = meshgrid(kx,ky);
ds=(pi/pxsize)^2*4;
% wave number
k = 2*pi/wavlen;

% circular convoluion via ffts
inputFT = fftshift(fft2(w_i));
CTF = exp(1i*dist*sqrt(k^2-KX.^2-KY.^2));
OTF=ifft2(ifftshift(conj(fftshift(fft2(CTF))).*fftshift(fft2(CTF))))./ds;
w_o = ifft2(ifftshift(inputFT.*OTF));

end

