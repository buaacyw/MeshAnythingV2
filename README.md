<p align="center">
  <h3 align="center"><strong>MeshAnything V2:<br> Artist-Created Mesh Generation<br>With Adjacent Mesh Tokenization</strong></h3>

<p align="center">
    <a href="https://buaacyw.github.io/">Yiwen Chen</a><sup>1</sup>,
    <a href="https://yikaiw.github.io/">Yikai Wang</a><sup>2</sup><span class="note">*</span>,
    <a href="https://github.com/Luo-Yihao">Yihao Luo</a><sup>3</sup>,
    <a href="https://thuwzy.github.io/">Zhengyi Wang</a><sup>2</sup>,
    <br>
    <a href="https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en">Zilong Chen</a><sup>2</sup>,
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml">Jun Zhu</a><sup>2</sup>,
    <a href="https://icoz69.github.io/">Chi Zhang</a><sup>4</sup><span class="note">*</span>,
    <a href="https://guosheng.github.io/">Guosheng Lin</a><sup>1</sup><span class="note">*</span>
    <br>
    <sup>*</sup>Corresponding authors.
    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University,
    <sup>2</sup>Tsinghua University,
    <br>
    <sup>3</sup>Imperial College London,
    <sup>4</sup>Westlake University
</p>



<div align="center">

<a href='https://arxiv.org/abs/2408.02555'><img src='https://img.shields.io/badge/arXiv-2408.02555-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://buaacyw.github.io/meshanything-v2/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/Yiwen-ntu/MeshAnythingV2/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/Yiwen-ntu/MeshAnythingV2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-HF-orange"></a>

</div>


<p align="center">
    <img src="demo/demo_video.gif" alt="Demo GIF" width="512px" />
</p>


## Contents
- [Contents](#contents)
- [Installation](#installation)
- [Usage](#usage)
- [Important Notes](#important-notes)
- [Acknowledgement](#acknowledgement)
- [BibTeX](#bibtex)

## Installation
Our environment has been tested on Ubuntu 22, CUDA 11.8 with A800.
1. Clone our repo and create conda environment
```
git clone https://github.com/buaacyw/MeshAnythingV2.git && cd MeshAnythingV2
conda create -n MeshAnythingV2 python==3.10.13 -y
conda activate MeshAnythingV2
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -U gradio
```

## Usage

### Implementation of Adjacent Mesh Tokenization and Detokenization
```
# We release our adjacent mesh tokenization implementation in adjacent_mesh_tokenization.py.
# For detokenization please check the function adjacent_detokenize in MeshAnything/models/meshanything_v2.py
python adjacent_mesh_tokenization.py
```


### For text/image to Artist-Create Mesh. We suggest using [Rodin](https://hyperhuman.deemos.com/rodin) to first achieve text or image to dense mesh. And then input the dense mesh to us.
```
# Put the output obj file of Rodin to rodin_result and using the following command to generate the Artist-Created Mesh.
# We suggest using the --mc flag to preprocess the input mesh with Marching Cubes first. This helps us to align the inference point cloud to our training domain.
python main.py --input_dir rodin_result --out_dir mesh_output --input_type mesh --mc
```

### Mesh Command line inference
#### Important Notes: If your mesh input is not produced by Marching Cubes, We suggest you to preprocess the mesh with Marching Cubes first (simply by adding --mc).
```
# folder input
python main.py --input_dir examples --out_dir mesh_output --input_type mesh

# single file input
python main.py --input_path examples/wand.obj --out_dir mesh_output --input_type mesh

# Preprocess with Marching Cubes first
python main.py --input_dir examples --out_dir mesh_output --input_type mesh --mc

# The mc resolution is default to be 128. For some delicate mesh, this resolution is not sufficient. Raise this resolution takes more time to preprocess but should achieve a better result.
# Change it by : --mc_level 7 -> 128 (2^7), --mc_level 8 -> 256 (2^8).
# 256 resolution Marching Cube example.
python main.py --input_dir examples --out_dir mesh_output --input_type mesh --mc --mc_level 8
```

### Point Cloud Command line inference
```
# Note: if you want to use your own point cloud, please make sure the normal is included.
# The file format should be a .npy file with shape (N, 6), where N is the number of points. The first 3 columns are the coordinates, and the last 3 columns are the normal.

# inference for folder
python main.py --input_dir pc_examples --out_dir pc_output --input_type pc_normal

# inference for single file
python main.py --input_path pc_examples/grenade.npy --out_dir pc_output --input_type pc_normal
```

### Local Gradio Demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>
```
python app.py
```

## Important Notes
- It takes about 8GB and 45s to generate a mesh on an A6000 GPU (depending on the face number of the generated mesh).
- The input mesh will be normalized to a unit bounding box. The up vector of the input mesh should be +Y for better results.
- Limited by computational resources, MeshAnything is trained on meshes with fewer than 1600 faces and cannot generate meshes with more than 1600 faces. The shape of the input mesh should be sharp enough; otherwise, it will be challenging to represent it with only 1600 faces. Thus, feed-forward 3D generation methods may often produce bad results due to insufficient shape quality. We suggest using results from 3D reconstruction, scanning, SDS-based method (like [DreamCraft3D](https://github.com/deepseek-ai/DreamCraft3D)) or [Rodin](https://hyperhuman.deemos.com/rodin) as the input of MeshAnything.
- Please refer to https://huggingface.co/spaces/Yiwen-ntu/MeshAnything/tree/main/examples for more examples.

## Acknowledgement

Our code is based on these wonderful repos:

* [MeshAnything](https://github.com/buaacyw/MeshAnything)
* [MeshGPT](https://nihalsid.github.io/mesh-gpt/)
* [meshgpt-pytorch](https://github.com/lucidrains/meshgpt-pytorch)
* [Michelangelo](https://github.com/NeuralCarver/Michelangelo)
* [transformers](https://github.com/huggingface/transformers)
* [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

## BibTeX
```
@misc{chen2024meshanythingv2artistcreatedmesh,
      title={MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization}, 
      author={Yiwen Chen and Yikai Wang and Yihao Luo and Zhengyi Wang and Zilong Chen and Jun Zhu and Chi Zhang and Guosheng Lin},
      year={2024},
      eprint={2408.02555},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02555}, 
}
```
