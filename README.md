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
- [ ] Official instructions of installation and usage of SketchRefiner.
- [ ] Testing code of SketchRefiner.
- [ ] The proposed sketch-based test protocol.
- [ ] Online demo of SketchRefiner.
- [ ] Pre-trained model weights.
- [ ] Training code of SketchRefiner.

# Prerequisites
To be implemented.

# Usage
To be implemented.

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

