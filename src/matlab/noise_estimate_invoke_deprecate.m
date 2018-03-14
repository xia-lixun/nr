function y = noise_estimate_invoke_deprecate(s, x)
% simple band noise energy estimator
% s: specificatoin
% x: spectrum frames in complex


nfft = s.feature.frame_length;
hop = s.feature.hop_length;
nframes = size(x,2);

Tau_be = s.noise_estimate.tau_be;
C_inc_dB = s.noise_estimate.c_inc_db; 
C_dec_dB = s.noise_estimate.c_dec_db; 
NoiseInit_dB = s.noise_estimate.noise_init_db;
MinNoiseNR_dB = s.noise_estimate.min_noise_db;


Alpha_be = exp((-5*hop)/(Tau_be * fs));
C_inc = 10^(hop*(C_inc_dB/20)/fs);
C_dec = 10^-(hop*(C_dec_dB/20)/fs);
BandEnergy_Mat = zeros(nfft/2+1,nframes+1);
BandNoise_Mat = 10^(NoiseInit_dB/20)*ones(nfft/2+1,nframes+1);
MinNoiseNR = 10^(MinNoiseNR_dB/20);



for k = 1:nframes
    BandEnergy_Mat(:,k+1) = Alpha_be * BandEnergy_Mat(:,k) + (1-Alpha_be) * abs(x(:,k));
    
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
y = BandNoise_Mat(:,2:end);
end