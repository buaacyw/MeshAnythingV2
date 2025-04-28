# Michelangelo

## [Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation](https://neuralcarver.github.io/michelangelo)<br/>
[Zibo Zhao](https://github.com/Maikouuu),
[Wen Liu](https://github.com/StevenLiuWen),
[Xin Chen](https://chenxin.tech/),
[Xianfang Zeng](https://github.com/Zzlongjuanfeng),
[Rui Wang](https://wrong.wang/),
[Pei Cheng](https://neuralcarver.github.io/michelangelo),
[Bin Fu](https://neuralcarver.github.io/michelangelo),
[Tao Chen](https://eetchen.github.io),
[Gang Yu](https://www.skicyyu.org),
[Shenghua Gao](https://sist.shanghaitech.edu.cn/sist_en/2020/0814/c7582a54772/page.htm)<br/>
### [Hugging Face Demo](https://huggingface.co/spaces/Maikou/Michelangelo) | [Project Page](https://neuralcarver.github.io/michelangelo/) | [Arxiv](https://arxiv.org/abs/2306.17115) | [Paper](https://openreview.net/pdf?id=xmxgMij3LY)<br/>

https://github.com/NeuralCarver/Michelangelo/assets/37449470/123bae2c-fbb1-4d63-bd13-0e300a550868

Visualization of the 3D shape produced by our framework, which splits into triplets with a conditional input on the left, a normal map in the middle, and a triangle mesh on the right. The generated 3D shapes semantically conform to the visual or textural conditional inputs.<br/>

## 🔆 Features
**Michelangelo** possesses three capabilities: 

1. Representing a shape into shape-image-text aligned space;
2. Image-conditioned Shape Generation;
3. Text-conditioned Shape Generation.

<details>
  <summary><b> Techniques </b></summary>

We present a novel _alignment-before-generation_ approach to tackle the challenging task of generating general 3D shapes based on 2D images or texts. Directly learning a conditional generative model from images or texts to 3D shapes is prone to producing inconsistent results with the conditions because 3D shapes have an additional dimension whose distribution significantly differs from that of 2D images and texts. To bridge the domain gap among the three modalities and facilitate multi-modal-conditioned 3D shape generation, we explore representing 3D shapes in a shape-image-text-aligned space. Our framework comprises two models: a Shape-Image-Text-Aligned Variational Auto-Encoder (SITA-VAE) and a conditional Aligned Shape Latent Diffusion Model (ASLDM). The former model encodes the 3D shapes into the shape latent space aligned to the image and text and reconstructs the fine-grained 3D neural fields corresponding to given shape embeddings via the transformer-based decoder. The latter model learns a probabilistic mapping function from the image or text space to the latent shape space. Our extensive experiments demonstrate that our proposed approach can generate higher-quality and more diverse 3D shapes that better semantically conform to the visual or textural conditional inputs, validating the effectiveness of the shape-image-text-aligned space for cross-modality 3D shape generation.

![newnetwork](https://github.com/NeuralCarver/Michelangelo/assets/16475892/d5231fb7-7768-45ee-92e1-3599a4c43a2c)
</details>

## 📰 News
- [2024/1/23] Set up the <a href="https://huggingface.co/spaces/Maikou/Michelangelo">Hugging Face Demo</a> and release the code
- [2023/09/22] **Michelangelo got accepted by NeurIPS 2023!**
- [2023/6/29] Upload paper and init project

## ⚙️ Setup

### Installation
Follow the command below to install the environment. We have tested the installation package on Tesla V100 and Tesla T4. 
```
git clone https://github.com/NeuralCarver/Michelangelo.git
cd Michelangelo
conda create --name Michelangelo python=3.9
conda activate Michelangelo 
pip install -r requirements.txt
```

### Checkpoints
Pleasae download weights from <a href="https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints">Hugging Face Model Space</a> and put it to root folder. We have also uploaded the weights related to CLIP to facilitate quick usage.

<details>
  <summary><b>  
    Tips for debugging configureation
  </b></summary>

- If something goes wrong in the environment configuration process unfortunately, the user may consider skipping those packages, such as pysdf, torch-cluster, and torch-scatter. These packages will not affect the execution of the commands we provide.
- If you encounter any issues while downloading CLIP, you can consider downloading it from [CLIP's Hugging Face page](https://huggingface.co/openai/clip-vit-large-patch14). Once the download is complete, remember to modify line [26](https://github.com/NeuralCarver/Michelangelo/blob/b53fa004cd4aeb0f4eb4d159ecec8489a4450dab/configs/text_cond_diffuser_asl/text-ASLDM-256.yaml#L26C1-L26C76) and line [34](https://github.com/NeuralCarver/Michelangelo/blob/b53fa004cd4aeb0f4eb4d159ecec8489a4450dab/configs/text_cond_diffuser_asl/text-ASLDM-256.yaml#L34) in the config file for providing correct path of CLIP.
- From [issue 6](https://github.com/NeuralCarver/Michelangelo/issues/6#issuecomment-1913513382). For Windows users, running wsl2 + ubuntu 22.04, will have issues. As discussed in [issue 786](https://github.com/microsoft/WSL/issues/8587) it is just a matter to add this in the .bashrc:
```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH.
```
</details>

## ⚡ Quick Start

### Inference

#### Reconstruction a 3D shape
```
./scripts/inference/reconstruction.sh
```

#### Image-conditioned shape generation
```
./scripts/inference/image2mesh.sh
```

#### Text-conditioned shape generation
```
./scripts/inference/text2mesh.sh
```

#### Simply run all the scripts
```
./scripts/infer.sh
```


## ❓ FAQ

## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@inproceedings{
zhao2023michelangelo,
title={Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation},
author={Zibo Zhao and Wen Liu and Xin Chen and Xianfang Zeng and Rui Wang and Pei Cheng and BIN FU and Tao Chen and Gang YU and Shenghua Gao},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=xmxgMij3LY}
}
```

## License

This code is distributed under an [GPL-3.0 license](LICENSE).

