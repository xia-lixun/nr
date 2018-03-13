module Mix
# management of wav files for machine learning projects
# lixun.xia@outlook.com
# 2017-10-16


import MAT
import WAV
import JSON

include("filesystem.jl")
include("neural.jl")
include("visual.jl")
include("feature.jl")




struct Specification

    seed::Int64    
    root_mix::String
    root_speech::String
    root_noise::String
    root_tensor::String

    sample_rate::Int64
    dbspl::Array{Float64,1}
    snr::Array{Float64,1}

    time_ratio::Float64
    split_ratio::Float64

    train_seconds::Float64
    test_seconds::Float64

    feature::Dict{String,Int64}
    noise_estimate::Dict{String, Float64}
    noise_groups::Array{Dict{String,Any},1}


    function Specification(path_json)
        s = JSON.parsefile(path_json)
        sum_percent = 0.0
        for i in s["noise_groups"]
            sum_percent += i["percent"]
        end
        assert(99.9 < sum_percent < 100.1)
        new(
            s["random_seed"],s["root_mix"],s["root_speech"],s["root_noise"],s["root_tensor"],
            s["sample_rate"],s["speech_level_db"],s["snr"],
            s["speech_noise_time_ratio"],s["train_test_split_ratio"],
            s["train_seconds"],s["test_seconds"],
            s["feature"],s["noise_estimate"],s["noise_groups"]
            )
    end
end


struct Layout

    noise_levels
    noise_keys
    noise_split

    speech_levels
    speech_keys
    speech_ratio  # the actual split ratio
    speech_point  # train-test split point

    function Layout(s::Specification)
        noise_levels = Dict(x["name"] => JSON.parsefile(joinpath(s.root_noise, x["name"], "level.json")) for x in s.noise_groups)
        noise_keys = Dict(x => shuffle([y for y in keys(noise_levels[x]["DATA"])]) for x in keys(noise_levels))
        noise_split = Dict(x => timesplit([noise_levels[x]["DATA"][y]["samples"] for y in noise_keys[x]], s.split_ratio) for x in keys(noise_keys))

        speech_levels = JSON.parsefile(joinpath(s.root_speech,"level.json"))
        speech_keys = shuffle([y for y in keys(speech_levels["DATA"])])
        speech_ratio, speech_point = timesplit([speech_levels["DATA"][x]["samples"] for x in speech_keys], s.split_ratio)

        new(noise_levels, noise_keys, noise_split, speech_levels, speech_keys, speech_ratio, speech_point)
    end
end



function generate_specification(path_mix::String, path_speech::String, path_noise::String)

    x = Array{Dict{String,Any},1}()
    a = Dict( 
        "root_mix" => path_mix,
        "root_speech" => path_speech,
        "root_noise" => path_noise,
        "root_tensor" => "\\172.20.20.20\\share\\",
        "sample_rate" => 16000,
        "speech_level_db" => [-22.0, -32.0, -42.0],
        "snr" => [20.0, 15.0, 10.0, 5.0, 0.0, -5.0],
        "speech_noise_time_ratio" => 0.6,
        "train_test_split_ratio" => 0.7,
        "train_seconds" => 1000,
        "test_seconds" => 1000,
        "random_seed" => 42,
        "feature" => Dict(
            "frame_length"=>1024, 
            "hop_length"=>256,
            "mel_bands" =>136, 
            "context_frames"=>31, 
            "nat_frames"=>31),
        "noise_estimate" => Dict(
            "tau_be"=>100e-3, 
            "c_inc_db"=>5.0,
            "c_dec_db"=>30.0,
            "noise_init_db"=> -40.0,
            "min_noise_db"=> -100.0),
        "noise_groups" => x
        )

    function foldersize(foldername)
        sz = 0
        for i in FileSystem.list(joinpath(a["root_noise"], foldername), t=".wav")
            sz += filesize(i)
        end
        return sz
    end

    foldernames = FileSystem.list(a["root_noise"])
    sz_list = zeros(Int64, length(foldernames))
    for (j,k) in enumerate(foldernames)
        sz_list[j] = foldersize(k)
    end
    for (j,k) in enumerate(foldernames)
        push!(x, Dict("name"=>k,"type"=>"stationary","percent"=>100 * (sz_list[j]/sum(sz_list))))
    end

    mkpath(a["root_mix"])
    open(joinpath(a["root_mix"],"specification-$(replace(replace("$(now())",":","-"),".","-")).json"),"w") do f
        write(f, JSON.json(a))
    end

    # generate initial checksum to trigger level update
    for i in a["noise_groups"]
        p = joinpath(a["root_noise"],i["name"])
        FileSystem.touch_checksum(p)
        info("checksum written to $p")
    end
    nothing
end


function build_level_index(path, rate)

    a = FileSystem.list(path, t=".wav")
    n = length(a)

    lpek = zeros(n)
    lrms = zeros(n)
    lmed = zeros(n)
    leng = zeros(Int64, n)

    # wav must be monochannel and fs==rate
    for (i,v) in enumerate(a)
        try
            x, fs = WAV.wavread(v)
            assert(fs == typeof(fs)(rate))
            assert(size(x,2) == 1)
            y = view(x,:,1)
            lpek[i] = maximum(abs.(y))
            lrms[i] = Fast.rms(y)
            lmed[i] = median(abs.(y))
            leng[i] = length(y)
        catch
            warn(v)
        end
    end

    index = joinpath(path, "index.level")
    writedlm(index, [a lpek lrms lmed leng], ',')
    info("index build to $index")
    nothing
end


function build_level_json(path, rate::Int64)

    a = FileSystem.list(path, t=".wav")
    d = Dict{String,Any}()
    d["DATA"] = Dict{String,Any}()

    lmin = typemax(Int64)
    lmax = 0
    lsum = 0

    # wav must be monochannel and fs==rate
    for i in a
        try
            x, fs = WAV.wavread(i)
            assert(typeof(rate)(fs) == rate)
            assert(size(x,2) == 1)

            y = view(x,:,1)
            n = length(y)
            d["DATA"][relpath(i, path)] = Dict("peak"=>maximum(abs.(y)), "rms"=>Fast.rms(y), "median"=>median(abs.(y)), "samples"=>n)
            n < lmin && (lmin = n)
            n > lmax && (lmax = n)
            lsum += n
        catch
            warn(i)
        end
    end
    d["META"] = Dict("len_min"=>lmin, "len_max"=>lmax, "len_sum"=>lsum, "sample_rate"=>rate)
    
    index = joinpath(path, "level.json")
    open(index, "w") do f
        write(f,JSON.json(d))
    end
    info("index written to $index")
    nothing
end



function merge_speech_json(path)
    # merge json of subdirs into its parent folder
    subfolders = readdir(path)
    united = Dict{String, Any}()
    united["DATA"] = Dict{String, Any}()
    united["META"] = Dict{String, Any}()

    united["META"]["sample_rate"] = 0
    united["META"]["len_max"] = 0
    united["META"]["len_min"] = 16000*3600*365
    united["META"]["len_sum"] = 0

    for i in subfolders
        partial = JSON.parsefile(joinpath(path, i, "level.json"))
        for j in partial["DATA"]
            united["DATA"][i * "/" * j[1]] = j[2]
        end
        partial["META"]["len_max"] > united["META"]["len_max"] && (united["META"]["len_max"] = partial["META"]["len_max"])
        partial["META"]["len_min"] < united["META"]["len_min"] && (united["META"]["len_min"] = partial["META"]["len_min"])
        united["META"]["len_sum"] += partial["META"]["len_sum"]
        united["META"]["sample_rate"] = partial["META"]["sample_rate"]
    end

    open(joinpath(path,"level.json"),"w") do f
        write(f, JSON.json(united))
    end
end





function wavread_safe(path)
    try
        x,fs = WAV.wavread(path)
    catch
        error("missing $path")
    end
end

function cyclic_extend(x::AbstractArray, n::Int64)
    x_extend = zeros(eltype(x), n)
    for (i,v) in enumerate(Iterators.cycle(x))
        x_extend[i] = v
        i == n && break
    end
    x_extend
end

function cyclic_extend!(x::AbstractArray, x_extend::AbstractArray)
    n = length(x_extend)
    for (i,v) in enumerate(Iterators.cycle(x))
        x_extend[i] = v
        i == n && break
    end
    nothing
end

function borders(partition)
    fin = cumsum(partition)
    beg = vcat(1, 1+fin[1:end-1])
    (beg,fin)
end

function timesplit(x, ratio)
    xs = cumsum(x)
    minval, offset = findmin(abs.(xs / xs[end] - ratio))
    y = xs[offset]/xs[end]
    (y, offset)
end








function wavgen(s::Specification, data::Layout; flag="train")
    # return: information that reconstructs source components
    # side-effect: write mixed wav files to /flag/wav/*.wav

    gain = Dict{String, Array{Float64,1}}()
    label = Dict{String, Array{Tuple{Int64, Int64},1}}()
    source = Dict{String, Tuple{String, String}}()

    root_mix_flag_wav = joinpath(s.root_mix, flag, "wav")
    rm(root_mix_flag_wav, force=true, recursive=true)
    mkpath(root_mix_flag_wav)

    time = (flag=="train")? s.train_seconds : s.test_seconds
    n_count = 1

    open(joinpath(s.root_mix,"history.log"),"a") do fid
        for cat in s.noise_groups

            group_samples = Int64(round(0.01cat["percent"] * time * s.sample_rate))
            name = cat["name"]
            group_samples_count = 0

            while group_samples_count < group_samples

                voice_spl_tt::Float64 = rand(s.dbspl)
                snr_tt::Float64 = rand(s.snr)
                if flag=="train" 
                    rn_voice_key = rand(view(data.speech_keys,1:data.speech_point))
                else 
                    rn_voice_key = rand(view(data.speech_keys,data.speech_point+1:length(data.speech_keys)))
                end
                if flag=="train" 
                    rn_noise_key = rand(view(data.noise_keys[name],1:data.noise_split[name][2]))
                else
                    rn_noise_key = rand(view(data.noise_keys[name],data.noise_split[name][2]+1:length(data.noise_keys[name])))
                end
                
                voice_wav::String = realpath(joinpath(s.root_speech, rn_voice_key))
                voice_lpk::Float64 = data.speech_levels["DATA"][rn_voice_key]["peak"]
                voice_spl::Float64 = data.speech_levels["DATA"][rn_voice_key]["dBrms"]
                voice_len::Int64 = data.speech_levels["DATA"][rn_voice_key]["samples"]

                block = data.noise_levels[name]
                noise_wav::String = realpath(joinpath(s.root_noise, name, rn_noise_key))
                noise_lpk::Float64 = block["DATA"][rn_noise_key]["peak"]
                noise_rms::Float64 = block["DATA"][rn_noise_key]["rms"]
                noise_med::Float64 = block["DATA"][rn_noise_key]["median"]
                noise_len::Int64 = block["DATA"][rn_noise_key]["samples"]

                # record the gains applied to speech and noise
                gain_ = zeros(2)

                # level speech to target
                x1,sr = wavread_safe(voice_wav)
                assert(typeof(s.sample_rate)(sr) == s.sample_rate)
                x = view(x1,:,1)

                g = 10^((voice_spl_tt-voice_spl)/20)
                if g * voice_lpk > 0.999
                    g = 0.999 / voice_lpk
                    voice_spl_tt = voice_spl + 20log10(g+eps())
                    write(fid,"[3] avoid clipping in speech [$(voice_wav)]\n")
                    write(fid,"[3] [$(voice_spl)] dB --> [$(voice_spl_tt)] dB\n")
                end
                x .= g .* x
                gain_[1] = g

                # get the random noise
                # random snr -> calculate noise level based on speech level and snr
                x2,sr = wavread_safe(noise_wav)
                assert(typeof(s.sample_rate)(sr) == s.sample_rate)
                u = view(x2,:,1)

                t = 10^((voice_spl_tt-snr_tt)/20)
                noisetype = cat["type"]
                if noisetype == "impulsive"
                    g = t / noise_lpk
                elseif noisetype == "stationary"
                    g = t / noise_rms
                elseif noisetype == "nonstationary"
                    g = t / noise_med
                else
                    error("wrong type in $(i["name"])")
                end
                if g * noise_lpk > 0.999
                    g = 0.999 / noise_lpk
                    write(fid, "[3] avoid clipping in noise [$(noise_wav)]\n")
                end
                u .= g .* u
                gain_[2] = g

                # voice-noise time ratio control
                noise_id = replace(relpath(noise_wav,s.root_noise), ['/', '\\'], "+")[1:end-4]
                voice_id = replace(relpath(voice_wav,s.root_speech), ['/', '\\'], "+")[1:end-4]

                pathout = joinpath(root_mix_flag_wav,"$(n_count)+$(noise_id)+$(voice_id)+$(voice_spl_tt)+$(snr_tt).wav")
                gain[pathout] = gain_
                source[pathout] = (voice_wav, noise_wav)
                η = voice_len/noise_len

                if η > s.time_ratio

                    noise_len_extend = Int64(round(voice_len / s.time_ratio))
                    u_extend = cyclic_extend(u, noise_len_extend)
                    r = rand(1:noise_len_extend-voice_len)
                    u_extend[r:r+voice_len-1] += x
                    WAV.wavwrite(u_extend, pathout, Fs=s.sample_rate)
                    label[pathout] = [(r, r+voice_len-1)]
                    group_samples_count += length(u_extend)

                elseif η < s.time_ratio

                    voice_len_tt = Int64(round(noise_len * s.time_ratio))
                    λ = voice_len_tt / voice_len   # 3.3|3.0
                    λr = floor(λ)                  # 3.0|3.0
                    λ1 = λr - 1.0                  # 2.0|2.0
                    λ2 = λ - λr + 1.0              # 1.3|1.0

                    voice_len_extend = Int64(round(voice_len * λ2))
                    x_extend = cyclic_extend(x, voice_len_extend)      # obs! length(x_extended) >= voice_len

                    ζ = Int64(round(noise_len / λ))
                    partition = zeros(Int64, Int64(λ1)+1)
                    for i = 1:Int64(λ1)
                        partition[i] = ζ
                    end
                    partition[end] = noise_len - Int64(λ1) * ζ
                    assert(partition[end] >= ζ)
                    shuffle!(partition)
                    (beg,fin) = borders(partition)

                    labelmark = Array{Tuple{Int64, Int64},1}()
                    for (i,v) in enumerate(partition)
                        if v > ζ
                            r = rand(beg[i] : fin[i]-voice_len_extend)
                            u[r:r+voice_len_extend-1] += x_extend
                            push!(labelmark,(r,r+voice_len_extend-1))
                        else
                            r = rand(beg[i] : fin[i]-voice_len)
                            u[r:r+voice_len-1] += x
                            push!(labelmark,(r,r+voice_len-1))
                        end
                    end
                    WAV.wavwrite(u, pathout, Fs=s.sample_rate)
                    label[pathout] = labelmark
                    group_samples_count += length(u)

                else
                    r = rand(1:noise_len-voice_len)
                    u[r:r+voice_len-1] += x
                    WAV.wavwrite(u, pathout, Fs=s.sample_rate)
                    label[pathout] = [(r, r+voice_len-1)]
                    group_samples_count += length(u)
                end
                n_count += 1
            end
            write(fid, "[3] [$(name)] processed\n")
        end
    end

    decomp_info = Dict(x => Dict("label"=>label[x], "gain"=>gain[x], "source"=>source[x]) for x in keys(label))
    open(joinpath(s.root_mix, flag, "info.json"),"w") do f
        write(f, JSON.json(decomp_info))
    end
    return decomp_info
end








function feature(s::Specification, decomp_info; flag="train")
    # return: nothing
    # side-effect: write ||spectrum|| and ratiomask to flag/spectrum/*.mat files in float64

    mel = Fast.Mel{Float64}(s.sample_rate, s.feature["frame_length"], s.feature["mel_bands"])

    function build_subdir(content)
        subdir = joinpath(s.root_mix, flag, content...)
        rm(subdir, force=true, recursive=true)
        mkpath(subdir)
        return subdir    
    end
    spectrum_dir = build_subdir(("spectrum",))
    oracle_mel_dir = build_subdir(("oracle", "mel"))
    oracle_dft_dir = build_subdir(("oracle", "dft"))
    decomp_dir = build_subdir(("decomposition",))


    for i in keys(decomp_info)

        x_mix, sr = WAV.wavread(i)
        assert(typeof(s.sample_rate)(sr) == s.sample_rate)
        x_mix = view(x_mix,:,1)

        x_voice,sr = WAV.wavread(decomp_info[i]["source"][1])
        assert(typeof(s.sample_rate)(sr) == s.sample_rate)
        x_voice = view(x_voice,:,1)
        x_voice .*= decomp_info[i]["gain"][1]

        x_purevoice= zeros(size(x_mix))
        for k in decomp_info[i]["label"]
            cyclic_extend!(x_voice, view(x_purevoice,k[1]:k[2]))
        end
        x_purenoise = x_mix - x_purevoice + rand(size(x_mix)) * (10^(-120/20))
        WAV.wavwrite(hcat(x_purevoice, x_purenoise), joinpath(decomp_dir, basename(i)),Fs=s.sample_rate)

        𝕏m, hm = Fast.stft2(x_mix, s.feature["frame_length"], s.feature["hop_length"], Fast.sqrthann)
        𝕏v, h = Fast.stft2(x_purevoice, s.feature["frame_length"], s.feature["hop_length"], Fast.sqrthann)
        𝕏n, h = Fast.stft2(x_purenoise, s.feature["frame_length"], s.feature["hop_length"], Fast.sqrthann)
        
        #@ ratiomask_dft_oracle = abs.(𝕏v) ./ (abs.(𝕏v) + abs.(𝕏n))    # ideal ratio mask
        #@ ratiomask_dft_oracle = real(𝕏v ./ (𝕏v + 𝕏n))                # phase sensitive mask
        ratiomask_dft_oracle = Float64.(abs.(𝕏v) .> abs.(𝕏n))          # binary mask

        ratiomask_mel_oracle = (mel.filter * ratiomask_dft_oracle) .* mel.weight
        magnitude_dft = abs.(𝕏m)
        
        magnitude_mel = log.((mel.filter * magnitude_dft) .* mel.weight +eps())
        #@ magnitude_mel = log.(mel.filter * magnitude_dft + eps())

        # noise estimate channel
        bandnoise = Fast.noise_estimate_invoke_deprecated(
            Fast.Frame1{Int64}(s.sample_rate, s.feature["frame_length"], s.feature["hop_length"], 0), 
            s.noise_estimate["tau_be"], 
            s.noise_estimate["c_inc_db"], 
            s.noise_estimate["c_dec_db"], 
            s.noise_estimate["noise_init_db"], 
            s.noise_estimate["min_noise_db"], 
            hcat(𝕏m, 𝕏m)
        )
        bandnoise = view(bandnoise,:,size(𝕏m,2)+1:2size(𝕏m,2))
        bandnoise_mel = log.((mel.filter * bandnoise) .* mel.weight + eps())

        MAT.matwrite(
            joinpath(spectrum_dir, basename(i[1:end-4]*".mat")), 
            Dict(
                # "ratiomask_dft"=>ratiomask_dft_oracle,
                # "spectrum_dft"=>magnitude_dft,
                "ratiomask_mel"=>ratiomask_mel_oracle,
                "spectrum_mel"=>magnitude_mel,
                "bandnoise_mel" => bandnoise_mel
                )
        )


        oracle_mel = Fast.stft2(
            𝕏m .* (mel.filter.' * ratiomask_mel_oracle), 
            hm, 
            s.feature["frame_length"], 
            s.feature["hop_length"], 
            Fast.sqrthann)
        WAV.wavwrite(2oracle_mel, joinpath(oracle_mel_dir,basename(i)), Fs=s.sample_rate)

        oracle_dft = Fast.stft2(
            𝕏m .* ratiomask_dft_oracle,
            hm, 
            s.feature["frame_length"], 
            s.feature["hop_length"], 
            Fast.sqrthann)
        WAV.wavwrite(2oracle_dft, joinpath(oracle_dft_dir,basename(i)), Fs=s.sample_rate)
    end
    nothing
end






function speech_weight(s::Specification)
    
    wav_list = FileSystem.list(s.root_speech, t=".wav")
    nframes = 0
    mel = Fast.Mel{Float64}(s.sample_rate, s.feature["frame_length"], s.feature["mel_bands"])
    y = zeros(size(mel.filter,1),1)

    for w in wav_list
        x, sr = WAV.wavread(w)
        x = view(x,:,1)
        𝕏, h = Fast.stft2(x, s.feature["frame_length"], s.feature["hop_length"], Fast.sqrthann)
        nframes += size(𝕏,2)
        y .+= sum(mel.filter * abs.(𝕏), 2)
    end
    y = y / nframes
    y = y / sum(y)
    MAT.matwrite(joinpath(s.root_speech, "spweight.mat"),Dict("ratio"=>Float32.(y)))
end


function statistics(s::Specification; flag = "train")
    # return: dictionary ["mu_spectrum"],["std_spectrum"],["mu_ratiomask"],["frames"]
    # side-effect: write dictionary aforementioned to /flag/statistics.mat

    spectrum_list = FileSystem.list(joinpath(s.root_mix, flag, "spectrum"), t=".mat")

    # detect dimensions
    spectrum_size = 0
    ratiomask_size = 0
    spectrum_frames = 0
    ratiomask_frames = 0

    for (k,i) in enumerate(spectrum_list)
        u = MAT.matread(i)
        spectrum_frames += size(u["spectrum_mel"],2)
        ratiomask_frames += size(u["ratiomask_mel"],2)
        if k == 1
            spectrum_size = size(u["spectrum_mel"],1)
            ratiomask_size = size(u["ratiomask_mel"],1)
        end
    end
    open(joinpath(s.root_mix,"history.log"),"a") do fid
        write(fid, "[5] $(flag) spectrum size = [$(spectrum_size),$(spectrum_frames)]\n")
        write(fid, "[5] $(flag) ratiomask size = [$(ratiomask_size),$(ratiomask_frames)]\n")
    end
    assert(spectrum_frames == ratiomask_frames)
    
    μ_spectrum = zeros(spectrum_size,1)
    σ_spectrum = zeros(spectrum_size,1)
    μ_ratiomask = zeros(ratiomask_size,1)


    # closure capture: spectrum_list
    function average!(feature::String, n::Int64, dest::Array{Float64,2})

        temp = zeros(BigFloat, size(dest,1), length(spectrum_list))
        for (j,i) in enumerate(spectrum_list)
            x = MAT.matread(i)
            for k = 1:size(dest,1)
                temp[k,j] = sum_kbn(view(x[feature],k,:))
            end
        end
        for k = 1:size(dest,1)
            dest[k] = sum_kbn(view(temp,k,:))/n
        end
        nothing
    end

    average!("spectrum_mel", spectrum_frames, μ_spectrum)
    average!("ratiomask_mel", ratiomask_frames, μ_ratiomask)
    open(joinpath(s.root_mix,"history.log"),"a") do fid
        write(fid, "[5] $(flag) global spectrum μ (dimentionless) = $(mean(μ_spectrum))\n")
        write(fid, "[5] $(flag) global ratiomask μ (dimentionless) = $(mean(μ_ratiomask))\n")
    end

    temp = zeros(BigFloat, spectrum_size, length(spectrum_list))
    for (j,i) in enumerate(spectrum_list)
        x = MAT.matread(i)
        for k = 1:spectrum_size
            temp[k,j] = sum_kbn((view(x["spectrum_mel"],k,:)-μ_spectrum[k]).^2)
        end
    end
    for k = 1:spectrum_size
        σ_spectrum[k] = sqrt(sum_kbn(view(temp,k,:))/(spectrum_frames-1))
    end
    open(joinpath(s.root_mix,"history.log"),"a") do fid
        write(fid, "[5] $(flag) global spectrum σ (dimentionless) = $(mean(σ_spectrum))\n")
    end

    statistics = Dict(
        "mu_spectrum"=>μ_spectrum, 
        "std_spectrum"=>σ_spectrum, 
        "mu_ratiomask"=>μ_ratiomask, 
        "frames"=>spectrum_frames
    )
    path_stat = joinpath(s.root_mix, flag, "statistics.mat")
    rm(path_stat, force=true)
    MAT.matwrite(path_stat, statistics)
    open(joinpath(s.root_mix,"history.log"),"a") do fid
        write(fid, "[5] $(flag) global statistics written to $(path_stat)\n")
    end
    return statistics
end





function tensor(s::Specification; flag="train")
    # return: nothing
    # side-effect: write tensors to root_tensor/flag/*.mat

    tensor_dir = joinpath(s.root_tensor, flag, "tensor")
    rm(tensor_dir, force=true, recursive=true)
    mkpath(tensor_dir)
    
    for i in FileSystem.list(joinpath(s.root_mix, flag, "spectrum"), t=".mat")

        data = MAT.matread(i)
        mixture = Float32.(Fast.sliding_aperture(data["spectrum_mel"], div(s.feature["context_frames"]-1,2), s.feature["nat_frames"]))
        noisech = Float32.(data["bandnoise_mel"])
        variable = vcat(mixture, noisech)
        label = Float32.(data["ratiomask_mel"])

        index = split(basename(i),"+")
        MAT.matwrite(joinpath(tensor_dir, "t_$(index[1]).mat"), Dict("variable"=>variable, "label"=>label))
    end
    nothing
end



















function ratiomask_inference(s::Specification, nn::Neural.Net{T}, mel::Fast.Mel{T}, 𝕏::Array{Complex{T},2}) where T <: AbstractFloat

    bandnoise = Fast.noise_estimate_invoke_deprecated(
        Fast.Frame1{Int64}(s.sample_rate, s.feature["frame_length"], s.feature["hop_length"], 0), 
        s.noise_estimate["tau_be"], 
        s.noise_estimate["c_inc_db"], 
        s.noise_estimate["c_dec_db"], 
        s.noise_estimate["noise_init_db"], 
        s.noise_estimate["min_noise_db"], 
        hcat(𝕏, 𝕏)
    )
    bandnoise = view(bandnoise,:,size(𝕏,2)+1:2size(𝕏,2))
    bandnoise_mel = log.((mel.filter * bandnoise) .* mel.weight + eps(T))

    mag_mel = log.((mel.filter * abs.(𝕏)) .* mel.weight + eps(T))
    mag_tensor = Fast.sliding_aperture(mag_mel, div(s.feature["context_frames"]-1,2), s.feature["nat_frames"])
    ratiomask_mel = Neural.feedforward(nn, vcat(mag_tensor,bandnoise_mel))
    ratiomask = mel.filter_t * ratiomask_mel
end



function wavform_reconstruct(s::Specification, nn::Neural.Net{Float32}, mel::Fast.Mel{Float32}, x::AbstractArray{Float32,1})
    # return: ratiomask inference
    # side-effect: write processed wav side-by-side to the original

    𝕏,h = Fast.stft2(x, s.feature["frame_length"], s.feature["hop_length"], Fast.sqrthann)
    ratiomask = ratiomask_inference(s, nn, mel, 𝕏)
    𝕏 .*= ratiomask
    y = Fast.stft2(𝕏, h, s.feature["frame_length"], s.feature["hop_length"], Fast.sqrthann)
    return (2y,ratiomask)
end



function process_dataset(specification_file::String, model_file::String, wav_dir::String)
    # return: ratiomask_infer
    # side-effect: write processed wav files to wav_dir/processed/

    s = Specification(specification_file)
    nn = Neural.Net{Float32}(model_file)
    mel = Fast.Mel{Float32}(s.sample_rate, s.feature["frame_length"], s.feature["mel_bands"])

    dir_out = joinpath(wav_dir, "processed")
    rm(dir_out, force=true, recursive=true)
    mkpath(dir_out)
    ratiomask_infer = Dict{String, Array{Float32,2}}()

    for i in FileSystem.list(wav_dir, t=".wav")
        x,sr = WAV.wavread(i)
        assert(typeof(s.sample_rate)(sr) == s.sample_rate)
        y,ratiomask_infer[i] = wavform_reconstruct(s, nn, mel, view(Float32.(x),:,1))
        WAV.wavwrite(y, joinpath(dir_out, basename(i)), Fs=s.sample_rate)
    end
    return ratiomask_infer
end



function sdr_benchmark(reference_dir::String, evaluation_dir::String; verbose=false)
    # file names must be identical in both folders
    # evaluation dir may contain a subset of the reference
    sdr = zeros(1,1)
    items = FileSystem.list(evaluation_dir, t=".wav")

    for i in items
        t,sr = WAV.wavread(joinpath(reference_dir, basename(i)))
        x,sr = WAV.wavread(i)
        sdr += Fast.signal_to_distortion_ratio(view(x,:,1), view(t,:,1))
        verbose && info(i)
    end
    return sdr/length(items)
end





# function benchmark(specification::String, bmerr::String)

#     s = JSON.parsefile(specification)   
#     m = div(s["feature"]["frame_length"],2)+1

#     file = HDF5.h5open(bmerr, "r")
#     bm = [(i, mean(read(file[i]),1), mean(read(file[i]),2)) for i in names(file)]
#     close(file)

#     # bm average over the whole batch
#     function gobal_average()
#         av = zeros(Float32, m)
#         for i in bm
#             av .+= vec(i[3])
#         end
#         av .= av ./ length(bm)
#     end

#     (gobal_average(),bm)

#     # sort!(bm, by=x->sum(x[3]), rev=true)                # worst case by all bins
#     # sort!(bm, by=x->sum(view(x[3],13:37,:)), rev=true)  # worst case by bin 13 to 37
#     # sort!(bm, by=x->maximum(x[2]), rev=true)            # worst case by highest dm deviation in frames
# end



# module
end







































# function ratiomask_inference(s::Specification, nn::Neural.Net{T}, 𝕏::Array{Complex{T},2}) where T <: AbstractFloat

#     mag_dft = log.(abs.(𝕏) + eps(T))
#     mag_tensor = Fast.sliding_aperture(mag_dft, div(s.feature["context_frames"]-1,2), s.feature["nat_frames"])
#     ratiomask = Neural.feedforward(nn, mag_tensor)
# end



# struct Stat{T <: AbstractFloat}
#     mu::Array{T,2}
#     std_rcp::Array{T,2}

#     function Stat{T}(path::String) where T <: AbstractFloat
#         stat = MAT.matread(path)
#         x = T.(stat["mu_spectrum"])
#         y = T.(stat["std_spectrum"])
#         y .= ones(T,size(y)) ./ (y .+ eps(T))
#         new(x, y)
#     end
# end



            # if mr[1] <= η <= mr[2]
            #     rd = rand(1:q-p)
            #     u[rd:rd+p-1] += x
            #     # clipping sample if over-range?
            #     path = joinpath(s["mix_root"],"wav","$(fcount)+$(nid)+$(sid)+1+1+$(sp)+$(sn).wav")
            #     WAV.wavwrite(u, path, Fs=fs)
            #     label[path] = [(rd, rd+p-1)]
            #     gain[path] = gvec
            # # η > mr[2] or η < mr[1]
            # else
            #     np = 1
            #     nq = 1
            #     while !(mr[1] <= η <= mr[2])
            #         η > mr[2] && (nq += 1)
            #         η < mr[1] && (np += 1)
            #         η = (np*p)/(nq*q)
            #     end
            #     path = joinpath(s["mix_root"],"wav","$(fcount)+$(nid)+$(sid)+$(np)+$(nq)+$(sp)+$(sn).wav")
            #     stamp = Array{Tuple{Int64, Int64},1}()

            #     u = repeat(u, outer=nq)
            #     pp = Int64(floor((nq*q)/np))
            #     for k = 0:np-1
            #         rd = k*pp+rand(1:pp-p)
            #         u[rd:rd+p-1] += x
            #         push!(stamp,(rd, rd+p-1))
            #     end
            #     WAV.wavwrite(u, path, Fs=fs)
            #     label[path] = stamp
            #     gain[path] = gvec
            # end
