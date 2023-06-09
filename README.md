# Towards Interactive Image Inpainting via Sketch Refinement
The official code implementation of "Towards Interactive Image Inpainting via Sketch Refinement".

[[Paper](https://arxiv.org/abs/2306.00407)] / [[Project](https://alonzoleeeooo.github.io/SketchRefiner/)] / [Test Protocol] / [Interactive Demo]

# Overview
![tissor](github_materials/teasor.jpg)
One tough problem of image inpainting is to restore complex structures in the corrupted regions. It motivates interactive image inpainting which leverages additional hints, e.g., sketches, to assist the inpainting process. Sketch is simple and intuitive to end users, but meanwhile has free forms with much randomness. Such randomness may confuse the inpainting models, and incur severe artifacts in completed images. To address this problem, we propose a two-stage image inpainting method termed SketchRefiner. In the first stage, we propose using a cross-correlation loss function to robustly calibrate and refine the user-provided sketches in a coarse-to-fine fashion. In the second stage, we learn to extract features from the abstracted sketches in a latent space and modulate the inpainting process. We also propose an algorithm to simulate real sketches automatically and build a test protocol to evaluate different methods under real applications. Experimental results on public datasets demonstrate that SketchRefiner effectively utilizes sketch information and eliminates the artifacts due to the free-form sketches. Our method consistently outperforms state-of-the-art baselines both qualitatively and quantitatively, meanwhile revealing great potential in real-world applications.

# To-Do Lists
<div align="center">
<img src="github_materials/star.jpg">
</div>

- Currently we are still working on collecting the codebase. We would open-source the codebase and the proposed dataset ASAP. Start the project to get notified!
- [x] Official instructions of installation and usage of SketchRefiner.
- [x] Testing code of SketchRefiner.
- [ ] The proposed sketch-based test protocol.
- [ ] Online demo of SketchRefiner.
- [ ] Pre-trained model weights.
- [x] Training code of SketchRefiner.

# Prerequisites
For installing the environment, you could execute the following scripts:
```bash
conda create -n sketchrefiner python=3.6
conda activate sketchrefiner
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
```

For utilizing the HRF loss to train the model, please download the pre-trained models provided by [LaMa](https://github.com/advimman/lama). You could download the model by executing the following command lines:
```bash
mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
```
Note that the `ade20k` folder is placed at `SIN_src/models/ade20k`.

# Usage
## Training
### 1. Train the Sketch Refinement Network (SRN)
Before training the inpainting network of SketchRefiner, you need to first train the proposed Sketch Refinement Network (SRN). We demonstrate an example command line as follows:

```bash
# train the Registration Module (RM)
python SRN_train.py
      --images /path/to/images
      --edges_prefix /path/to/edge
      --output /path/to/output/dir
      
# train the Enhancement Module (EM)
python SRN_train.py
      --images /path/to/images/
      --edges_prefix /path/to/edge/
      --output /path/to/output/dir
      --train_EM
      --RM_checkpoint /path/to/model/weights/of/RM
```

If you need to evaluate the model during training, make sure you assign `val_interval > 0` and configure the paths of `images_val`, `masks_val`, `sketches_prefix_val`, `edges_prefix_val`.

### 2. Train the Sketch-modulated Inpainting Network (SIN)
After the SRN is trained, you could start training the SIN by executing:
```bash
python SIN_train.py
      --config_path /path/to/config
      --GPU_ids gpu_id
```
Make sure you modify corresponding lines of the configuration file with the paths of training data. We put an example configuration file in `SIN_src/configs/example.yml`.

## Inference
### 3. Refine the Input Sketches
During inference, you need to first refine the input sketches and save them locally by running:
```bash
python SRN_test.py
      --images /path/to/test/source/images
      --masks /path/to/test/masks
      --edge_prefix /path/to/detected/edges
      --sketch_prefix /path/to/input/sketches
      --output /path/to/output/dir
      --RM_checkpoint /path/to/RM/checkpoint
      --EM_checkpoint /path/to/EM/checkpoint
```
The refined sketches would be saved at the assigned `output` path.

### 4. Inpaint the Corrupted Images with Refined Sketches
Now you could inpaint the corrupted images with the refined sketches from SRN, by running:
```bash
python SIN_test.py
      --config_path /path/to/config/path
      --GPU_ids gpu_id
      --images /path/to/source/images
      --masks /path/to/masks
      --edges /path/to/detected/edges
      --sketches /path/to/input/sketches
      --refined_sketches /path/to/refined/sketches
      --output /path/to/output/dir/
      --checkpoint /path/to/model/weights/of/SIM
      --num_samples maximum_samples
```
Edges are detected using bdcn, you could refer to their code in [here](https://github.com/pkuCactus/BDCN).

# Qualitative Comparisons
Qualitative comparison on CelebA-HQ dataset.
![celebahq](github_materials/celebahq.jpg)

Qualitative comparison on Places2 dataset.
![places](github_materials/places.jpg)

# Sketch-Based Test Protocol
We propose a sketch-based test protocol to promote further researches upon the studied task. The masks and sketches are manually annotated on iPad with an apple pencil.
![realworld](github_materials/realworld.jpg)

# More Results
More results of face editing and object removal are illustrated in the following figure.
![more_results](github_materials/more_results.jpg)

# License
This work is licensed under MIT license. See the [LICENSE](LICENSE) for details.

# Citation
If you find our work is enlightening or the proposed dataset is useful to you, please cite our paper.
```tex
@misc{liu2023interactive,
      title={Towards Interactive Image Inpainting via Sketch Refinement}, 
      author={Chang Liu and Shunxin Xu and Jialun Peng and Kaidong Zhang and Dong Liu},
      year={2023},
      eprint={2306.00407},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

