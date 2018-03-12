% simple band noise energy estimator
close all; clear all; clc;

nfft = 1024;
hop = nfft/4;


[y,fs] = audioread('D:\\1-Workspace\\test\\wav\\12+Accelerating, revving, vroom+m07q2z82+eyFPHlybqDg+TIMIT+dr5+fdtd0+sx121+-22.0+-5.0.wav');
y = [y;y;y];

win = sqrt(hann( nfft, 'periodic')); 
ys = stft2(y.',nfft,hop,0,win);
nframes = size(ys,2);






Tau_be = 100e-3;
C_inc_dB = 5;%30;%1;%10;%4;%
C_dec_dB = 30;%60;%6;24;%
NoiseInit_dB = -40;%-30;%-20;%
MinNoiseNR_dB = -100;%-60;%-50;%


Alpha_be = exp((-5*hop)/(Tau_be * fs));
C_inc = 10^((hop*(C_inc_dB)/20)/fs);
C_dec = 10^-(hop*(C_dec_dB/20)/fs);
BandEnergy_Mat = zeros(nfft/2+1,nframes+1);
BandNoise_Mat = 10^(NoiseInit_dB/20)*ones(nfft/2+1,nframes+1);
MinNoiseNR = 10^(MinNoiseNR_dB/20);




for k = 1:nframes
    BandEnergy_Mat(:,k+1) = Alpha_be * BandEnergy_Mat(:,k) + (1-Alpha_be) * abs(ys(:,k));
    
    for n = 1:nfft/2+1 
        if BandEnergy_Mat(n,k+1) > BandNoise_Mat(n,k)
            BandNoise_Mat(n,k+1) = C_inc * BandNoise_Mat(n,k);
        else
            BandNoise_Mat(n,k+1) = C_dec * BandNoise_Mat(n,k);
        end
        if BandNoise_Mat(n,k+1) < MinNoiseNR
            BandNoise_Mat(n,k+1) = MinNoiseNR;
        end
    end
end

simple_heatmap(BandNoise_Mat)





