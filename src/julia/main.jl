
include("mix.jl")








function build(path_specification)

    # 0
    s = Mix.Specification(path_specification)
    open(joinpath(s.root_mix,"history.log"),"a") do fid
        write(fid, "[0] [$(path_specification)]\n")
        write(fid, "[0] [$(now())]\n")
    end
    srand(s.seed)
    !isdir(s.root_noise) && error("noise depot doesn't exist")
    !isdir(s.root_speech) && error("speech depot doesn't exist")
    mkpath(s.root_mix)

    # 1
    for i in s.noise_groups
        path = joinpath(s.root_noise,i["name"])
        if !Mix.FileSystem.verify_checksum(path)
            info("checksum mismatch [$(i["name"])], updating level statistics")
            Mix.build_level_json(path, s.sample_rate)
            Mix.FileSystem.update_checksum(path)
        end
    end
    if !Mix.FileSystem.verify_checksum(s.root_speech)
        info("[todo - calculate speech level online]")
        Mix.FileSystem.update_checksum(s.root_speech)
    end

    # 2
    data = Mix.Layout(s)
    info("data layout formed")

    # 3
    train_info = Mix.wavgen(s, data)
    test_info = Mix.wavgen(s, data, flag="test")
    info("time series mixed")
    
    # 4
    Mix.feature(s, train_info)
    Mix.feature(s, test_info, flag="test")
    info("feature extracted")

    # 5
    train_stat = Mix.statistics(s)
    test_stat = Mix.statistics(s, flag="test")
    info("total frames  [train,test] = [$(train_stat["frames"]),$(test_stat["frames"])]")
    info("mean spectrum [train,test] = [$(mean(train_stat["mu_spectrum"])), $(mean(test_stat["mu_spectrum"]))]")
    info("mean ratiomsk [train,test] = [$(mean(train_stat["mu_ratiomask"])), $(mean(test_stat["mu_ratiomask"]))]")

    # 6
    sdr_oracle_dft = Mix.sdr_benchmark(joinpath(s.root_mix, "test", "decomposition"), joinpath(s.root_mix, "test", "oracle", "dft"))
    sdr_oracle_mel = Mix.sdr_benchmark(joinpath(s.root_mix, "test", "decomposition"), joinpath(s.root_mix, "test", "oracle", "mel"))
    open(joinpath(s.root_mix,"history.log"),"a") do fid
        write(fid, "[6] Oracle SDR DFT($(s.feature["frame_length"])) = $(sdr_oracle_dft) dB\n")
        write(fid, "[6] Oracle SDR Mel($(s.feature["mel_bands"])) = $(sdr_oracle_mel) dB\n")
    end
    info("Oracle SDR DFT($(s.feature["frame_length"])) = $(sdr_oracle_dft) dB")
    info("Oracle SDR Mel($(s.feature["mel_bands"])) = $(sdr_oracle_mel) dB")

    # 7
    Mix.tensor(s)
    Mix.tensor(s, flag="test")
    open(joinpath(s.root_mix,"history.log"),"a") do fid
        write(fid, "[7] tensors created\n")
    end
    info("tensors created")
    nothing
end
