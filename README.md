# REMNet: Recurrent Evolution Memory-Aware Network for Accurate Long-Term Weather Radar Echo Extrapolation
The official PyTorch implementation of [REMNet](https://ieeexplore.ieee.org/document/9856702?source=authoralert) (IEEE TGRS, 2022).

Authors: [Jinrui Jing](https://www.researchgate.net/profile/Jinrui-Jing), [Qian Li](https://www.researchgate.net/profile/Qian-Li-192), [Leiming Ma](https://www.researchgate.net/profile/Lei-Ming-Ma), Lei Chen, [Lei Ding](https://www.researchgate.net/profile/Lei-Ding-26)

## Paper Abstract
Weather radar echo extrapolation, which predicts future echoes based on historical observations, is one of the complicated spatial–temporal sequence prediction tasks and plays a prominent role in severe convection and precipitation nowcasting. However, existing extrapolation methods mainly focus on a defective echo-motion extrapolation paradigm based on finite observational dynamics, neglecting that the actual echo sequence has a more complicated evolution process that contains both nonlinear motions and the lifecycle from initiation to decay, resulting in poor prediction precision and limited application ability. To complement this paradigm, we propose to incorporate a novel long-term evolution regularity memory (LERM) module into the network, which can memorize long-term echo-evolution regularities during training and be recalled for guiding extrapolation. Moreover, to resolve the blurry prediction problem and improve forecast accuracy, we also adopt a coarse–fine hierarchical extrapolation strategy and compositive loss function. We separate the extrapolation task into coarse and fine two levels which can reduce the downsampling loss and retain echo fine details. Except for the average reconstruction loss, we additionally employ adversarial loss and perceptual similarity loss to further improve the visual quality. Experimental results from two real radar echo datasets demonstrate the effectiveness of our methodology and show that it can accurately extrapolate the echo evolution while ensuring the echo details are realistic enough, even for the long term. Our method can further be improved in the future by integrating multimodal radar variables or introducing certain domain prior knowledge of physical mechanisms. It can also be applied to other spatial–temporal sequence prediction tasks, such as the prediction of satellite cloud images and wind field figures.

## Setup
1. PyTorch >= 1.6.0
2. Anaconda, cuda, and cudnn are recommended
3. Other required python libraries: PIL, torchvision, tensorboard, skimage, tqdm, xlwt, matplotlib
4. Preparing the two radar echo datasets: [HKO-7](https://github.com/sxjscience/HKO-7), [Shanghai-2020](https://doi.org/10.5281/zenodo.7251972)
5. Set the right dataset root in <code>configs.py</code>
6. Python run the <code>HKO_7_preprocessing.py</code> and <code>Shanghai_2020_preprocessing.py</code> script in the <code>./datasets</code> folder to preprocess the two datasets before training

## Training
Run the <code>train_hko_7.py</code> or <code>train_shanghai_2020.py</code> script to train the model on two datasets respectively
<pre><code>$ python train_hko_7.py</code></pre>
<pre><code>$ python train_shanghai_2020.py</code></pre>

You can also change the default training settings in <code>configs.py</code>, such as the <code>device_ids</code> or <code>train_batch_size</code>. The training task can be distributed on multi-GPUs. In our work, the train batch size is set to 8 and can be distributed on 4 RTX 2080 Ti GPUs, each with 12 GB of memory.

The training log files will be stored in the <code>logdir</code> folder, you can open <code>tensorboard</code> website to visualize them.

The trained model will be saved in the <code>checkpoints</code> folder automatically and periodically (controlled by the <code>model_save_fre</code>).

## Test
To test the pretrained model on test set and samples, you can run
<pre><code>$ python test_hko_7.py</code></pre>
<pre><code>$ python test_shanghai_2020.py</code></pre>

Our pretrained model can be downloaded from link [HKO-7](https://k00.fr/ji3nlvjy) and [Shanghai-2020](https://k00.fr/py2vkihw), please put them into the <code>checkpoints</code> folder to test.

We have provided serveral test samples in <code>test_samples</code> folder, you can also add yours following the same pattern. The test results will be saved in <code>test_results</code> folder.

If you only want to test the model on test samples, not on test set quantitatively, please test with <code>test_samples_only=True</code>.

We have provided some test samples gifs in <code>figures</code> folder.

## Citation
When using any parts of the Project or the Paper in your work, please cite the following paper:
<pre><code>@InProceedings{Jing_2022_IEEE TGRS, 
  author = {Jing, Jinrui and Li, Qian and Ma, Leiming and Chen, Lei and Ding, Lei}, 
  title = {REMNet: Recurrent Evolution Memory-Aware Network for Accurate Long-Term Weather Radar Echo Extrapolation}, 
  journal = {IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)}, 
  year = {2022},
}</code></pre>
