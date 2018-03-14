function inference(s, model_path, wav_path, path_output)
    
    %path_output = fullfile(wav_path, 'processed');
    if ~exist(path_output, 'dir')
        mkdir(path_output);
    end
    delete(fullfile(path_output, '*.wav'));
    
    model = loadmodel(model_path);
    todo =  dir([wav_path '/**/*.wav']);

    nmel = s.feature.mel_filter_banks;
    nfft = s.feature.frame_length;
    hop = s.feature.hop_length;
    win = sqrt(hann(nfft,'periodic'));
    mel = filter_banks(s.sample_rate, nfft, nmel, 0, s.sample_rate/2);
    mel_weight = sum(mel,2);
    
    for i = 1:length(todo)
        
        path = fullfile(todo(i).folder, todo(i).name);
        [x,fs] = audioread(path);
        assert(fs == s.sample_rate)
        
        spectrum = stft2(x.', nfft, hop, 0, win);
        tensor = sliding(log((mel * abs(spectrum)) ./ mel_weight + eps), (s.feature.context_span-1)/2, s.feature.nat_frames);
        
        % noise estimate channel
        bandnoise = noise_estimate_invoke_deprecate(s, [spectrum spectrum]);
        bandnoise = bandnoise(:,size(spectrum,2)+1:2*size(spectrum,2));
        bandnoise_mel = log((mel * bandnoise) ./ mel_weight + eps);
        
        ratiomask_mel = feed_forward(model, [tensor; bandnoise_mel]);
        
        % reconstruct based on bm estimate
        model_reconstruct = (mel.' * ratiomask_mel) .* spectrum;
        scale = 2;
        speech_recovered = scale * stft2(model_reconstruct, nfft, hop, 0, win);
        audiowrite(fullfile(path_output, todo(i).name), speech_recovered, s.sample_rate, 'BitsPerSample', 32);
    end
    
end




function m = loadmodel(model_path)
    
    load(model_path,...
    'param_1','param_2',...
    'param_3','param_4',...
    'param_5','param_6',...
    'param_7','param_8',...
    'stats_bn1_mean', 'stats_bn1_var',...
    'stats_bn2_mean', 'stats_bn2_var',...
    'stats_bn3_mean', 'stats_bn3_var');

    model.nn(1).w = param_1;
    model.nn(1).b = param_2.';
    model.nn(2).w = param_3;
    model.nn(2).b = param_4.';
    model.nn(3).w = param_5;
    model.nn(3).b = param_6.';
    model.nn(4).w = param_7;
    model.nn(4).b = param_8.';
    
    model.nn(1).mu = stats_bn1_mean.';
    model.nn(1).std = sqrt(stats_bn1_var.' + 1e-5);
    model.nn(2).mu = stats_bn2_mean.';
    model.nn(2).std = sqrt(stats_bn2_var.' + 1e-5);
    model.nn(3).mu = stats_bn3_mean.';
    model.nn(3).std = sqrt(stats_bn3_var.' + 1e-5);    
    
    m = model;
end




function y = feed_forward(model, x)
    
    layers = length(model.nn);
    act = sigmoid(model.nn(1).w * x + model.nn(1).b);
    act = (act - model.nn(1).mu) ./ model.nn(1).std;
    for j = 2:layers-1
        act = sigmoid(model.nn(j).w * act + model.nn(j).b);
        act = (act - model.nn(j).mu) ./ model.nn(j).std;
    end
    y = sigmoid(model.nn(layers).w * act + model.nn(layers).b);
end


function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
