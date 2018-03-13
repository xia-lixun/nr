# Single-Channel Noise Reduction Roadmap
>1. Tensor in log-mel domain. vs. linear scale. 
>2. Add additional memories to T7910: 8 slots for each CPU socket. Full for CPU0; Empty for CPU1. Need another CPU to use the rest of the dimm slots? Supported max capacity of each dimm?
>3. add QUT and TUT noise database: examine the quality as no-speech contained noise samples. TUT contains obvious speech, cleanup needed. QUT contains obvious speech, cleanup needed.

## Cost Function
* **SA approach**: Phase-sensitive object function? Now we assume cost function f(bm, bm_hat) where f() could be MMSE or Crossentropy. Change the cost to _signal approximation_: ||y*bm_hat-x|| where y is the complex spectrum of the distorted sample, and x is the complex spectrum of the clean speech.
* with the convergence of the SA, conbine MA and SA for a new training framework 

## Model Architecture
* DNN
* CNN
* LSTM-DNN

## Data Soundness
### Speech and noise quality 
* remove blank in clean speech algorithmically?
* remove remaining speeches in the noise examples.
* how to extract high quality speech corpus from dataset like common-voice?

### Database management
* groups with fine-granularity turns out to be cumbersome in common use cases (imagine assigning percent/type to hundreds of noise categories...). Merge them into more abstract groups? Or devise an method to allow easy configuration in specification.
* examples in each noise group may belong to different type (stationary/impulsive/non-stationary), come up with a method to determine its type algorithmically. 

## Model Training Strategies
* Experiment with small scale dataset. Definition of "small scale": those which fit in memory. This is for model/parameter tuning.
* Large scale dataset reside on hard drives. Build efficient pipeline to process the data in reasonable amount of time. Definition of efficient: average utility of GPU be above 70%! This is for model release.

## Design
* system divided into functional blocks
* functional blocks communicates via file, return value is only for in-workplace inspection. This allows functional blocks as plugin-in.


## Deliverable
* core dataset of noise
* training scripts






# Appendix






## 2018-02-27
0. add cost of training calculation to training scripts
1. train baseline of v1.1; calculate SDR 
2. use 1024/mel point upon v1.1; calculate SDR
3. update model of the better one to Batch Normalization in TF
4. export params of BN layers
5. insert observations of the activation statistics
6. add more layers, go deep

## 2018-03-07
0. OK. loss of train/test both online
1. OK. SDR = 12.5
2. OK. 1024 works better than 512. (a)longer viewer scope in time domain; (b)mixing to longer hours as frames halves; (c)better resolution in mel bins=136
3. OK. Moved to Pytorch framework. (a)easier for individual dev in almost every respect
4. OK. In Pytorch framework everything is easier to access. BN after nonlinear activation way better than before activation!
5. WORKING..
6. OK. with BN and other techniques going deep is no problem. But so far we fix the model structure in order to tune feature/loss etc.

7. Investigate MSE loss based on the current optimal hyper params {block=1024, BN after activation, 3-layer FC-2048}
8. Incorporate phase information to loss function: multiple-object optimization

## 2018-03-08
1. introduce phase sensitive mask (psm): Re(s/y) = |s|/|y|cos(theta) where theta = theta_s - theta_y
2. oracle performance: DFT bins irm/psm = 18dB/23.7dB; Mel bins irm/psm = 17dB/21dB
3. matlab demo updated accordingly
3. saw hard in convergence due to fluctuations in psm? => sigmoid transformation       sdr = 9.5dB, not good!
                                                       => Re(y) as input to the net    sdr = ? loss convergence too high! abandon!
4. revert to irm method w/o phase information

5. try binary mask: -6dB SNR oracle performance seems higher than irm? confirmed! bm DFT/Mel 20.4/19.1                   
6. small dataset test: historical highest 13.2dB SDR!

7. removing simple noise-aware training: minimal loss unchanged, but converging progress fluctuated, indicating sharp minima(we prefer wide and smooth minima for better generalisation -> indicates necessity of noise-aware features for shallow nets!) 
        => use Markus's noise estimate? ...
        => average over context_frames? so far gives best results! SDR = 13.5dB! converging progress smooth and little dithering in equillibrium. <milestone>
        => median of context frames? not good, bad converging progress, showing fluctuation in validation at early steps.
        => model-stacking for inference? no good, SDR drops by 0.003dB
        => detection/approximation network? => 13.5 -> 13.6dB. 

4. Noise estimate from invoke project
5. Widen the view scope: no, 32-frame is good


6. LSTM-DNN or CNN?, RCNN
7. ask VASU for denoising performance


## 2018-03-12
0. update pytorch version on fx8310
1. validate minor update of 2018-03-11 on HISUPN071

1. repeat "devmat" commit on date 2018-03-08
2. incorporate invoke noise estimate

## 2018-03-13
0. apply mel.weight to spectrum helps! 13.2 -> 13.5dB
1. introduce noise estimate channel: invoke deprecated. training/validation loss converge fast and smooth!
                => 13.5 => 13.7dB SDR in validation. convergence smooth and faster. smaller train/test gap
2. try hyperbolic tan tanh=2sigmoid-1 as activation. NOT OK! Revert to sigmoid!
3. add layers when test error stops shrinking. OK? validation loss even lower, but convergence fluctuated, need more data?
4. reshape the weight init. OK
5. try Adam with adaptive learning rate