% data mixing scripts for selective noise reduction
% lixun.xia2@harman.com
% 2018-01-16
function s = specification()

specification.root = 'D:\7-Workspace\';
specification.noise = 'D:\oak\noise_google_audio_set\no_speech\';
specification.speech = 'D:\oak\speech_timit_lp7_matlab\';

specification.speech_noise_time_ratio = 0.6;
specification.split_ratio_for_training = 0.7;   % this is also time ratio
specification.speech_level_db = [-22.0 -32.0 -42.0];
specification.snr = [20.0 15.0 10.0 5.0 0.0 -5.0];

specification.training_seconds = 14400;
specification.testing_seconds = 7200;
specification.sample_rate = 16000;
specification.random_seed = 42;

specification.feature.frame_length = 1024;
specification.feature.hop_length = 256;
specification.feature.context_span = 31;
specification.feature.nat_frames = 31;
specification.feature.mel_filter_banks = 136;

specification.noise_estimate.tau_be = 100e-3;
specification.noise_estimate.c_inc_db = 5.0;  %30;%1;%10;%4;%
specification.noise_estimate.c_dec_db = 30.0; %60;%6;24;%
specification.noise_estimate.noise_init_db = -40.0; %-30;%-20;%
specification.noise_estimate.min_noise_db = -100.0; %-60;%-50;%


%------------------------noise database----------------------------
specification.noisegroup(1).name = 'Accelerating, revving, vroom';
specification.noisegroup(1).percent = 10.0;
specification.noisegroup(1).type = 'stationary';

specification.noisegroup(2).name = 'Air brake';
specification.noisegroup(2).percent = 10.0;
specification.noisegroup(2).type = 'stationary';

specification.noisegroup(3).name = 'Air conditioning';
specification.noisegroup(3).percent = 10.0;
specification.noisegroup(3).type = 'stationary';

specification.noisegroup(4).name = 'Air horn, truck horn';
specification.noisegroup(4).percent = 10.0;
specification.noisegroup(4).type = 'stationary';

specification.noisegroup(5).name = 'Aircraft';
specification.noisegroup(5).percent = 10.0;
specification.noisegroup(5).type = 'stationary';

specification.noisegroup(6).name = 'Aircraft engine';
specification.noisegroup(6).percent = 10.0;
specification.noisegroup(6).type = 'stationary';

specification.noisegroup(7).name = 'Alarm';
specification.noisegroup(7).percent = 10.0;
specification.noisegroup(7).type = 'stationary';

specification.noisegroup(8).name = 'Alarm clock';
specification.noisegroup(8).percent = 10.0;
specification.noisegroup(8).type = 'stationary';

specification.noisegroup(9).name = 'Ambulance (siren)';
specification.noisegroup(9).percent = 10.0;
specification.noisegroup(9).type = 'stationary';

specification.noisegroup(10).name = 'Applause';
specification.noisegroup(10).percent = 10.0;
specification.noisegroup(10).type = 'stationary';

s = specification;
end