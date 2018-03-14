% data mixing scripts for dnn speech enhancement
% lixun.xia2@harman.com
% v1.1 2018-01-16
% v1.2 2018-03-14
function [sdr_oracle_dft, sdr_oracle_mel] = mix()

    s = specification();
    data = index(s);
    
    [train_label, train_log] = generate_wav(s, data, 'training');
    [test_label, test_log] = generate_wav(s, data, 'testing');
    
    save2json(train_label, fullfile(s.root, 'training', 'info.json'));
    save2json(train_log, fullfile(s.root, 'training', 'log.json'));
    save2json(test_label, fullfile(s.root, 'testing', 'info.json'));
    save2json(test_log, fullfile(s.root, 'testing', 'log.json'));
    
    train_frames = generate_feature(s, train_label, 'training');
    test_frames = generate_feature(s, test_label, 'testing');
    
    [mu_bm_train, mu_spec_train, std_bm_train, std_spec_train] = statistics(s, train_frames, 'training');
    [mu_bm_test, mu_spec_test, std_bm_test, std_spec_test] = statistics(s, test_frames, 'testing');
    figure; hold on; grid on; plot(mu_bm_train); plot(mu_bm_test);
    figure; hold on; grid on; plot(mu_spec_train); plot(mu_spec_test);
    
    sdr_oracle_dft = sdr_benchmark(fullfile(s.root,'training', 'oracle', 'dft'), fullfile(s.root, 'training', 'decomposition'));
    sdr_oracle_mel = sdr_benchmark(fullfile(s.root,'training', 'oracle', 'mel'), fullfile(s.root, 'training', 'decomposition'));
    
    tensor(s, 'training');
    tensor(s, 'testing');
    
end









function m = index(s)
% build speech/noise data layout for mixing.

rng(s.random_seed);
tmp = rand(100);
clear tmp;

% populate noise group folders and calculate levels
for i = 1:length(s.noisegroup)
    
    path_group = fullfile(s.noise, s.noisegroup(i).name);
    group =  dir([path_group '/**/*.wav']);
    % populate noise examples and shuffle the sequence within each group
    for j = 1:length(group)    
        layout.noise(i).path(j) = cellstr(fullfile(group(j).folder, group(j).name));  
    end
    layout.noise(i).path = layout.noise(i).path(randperm(length(group)));
    
    % calculate the levels of the shuffled examples
    for j = 1:length(group)
        
        [x, fs] = audioread(layout.noise(i).path{j});
        assert(fs == s.sample_rate)
        assert(size(x,2) == 1)
        y = abs(x);
        layout.noise(i).level_max(j) = max(y);
        layout.noise(i).level_rms(j) = rms(x);
        layout.noise(i).level_med(j) = median(y);
        layout.noise(i).length(j) = size(y,1);
    end
end


% load clean speech level info
fid = fopen(fullfile(s.speech, 'index.level'));
c = textscan(fid, '%s %f %f %f', 'Delimiter',',', 'CommentStyle','#');
fclose(fid);

shuffle = randperm(length(c{1}));
layout.speech.path = c{1};
layout.speech.level_max = c{2};
layout.speech.level_dbrms = c{3};
layout.speech.length = c{4};

layout.speech.path = layout.speech.path(shuffle);
layout.speech.level_max = layout.speech.level_max(shuffle);
layout.speech.level_dbrms = layout.speech.level_dbrms(shuffle);
layout.speech.length = layout.speech.length(shuffle);



m = layout;
end



function save2json(object, path)
% save struct object to json
    fid = fopen(path,'wt');
    fprintf(fid, jsonencode(object));
    fclose(fid);
end



