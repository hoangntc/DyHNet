# Installation

## Create env
```conda create -n pygnn python=3.9 pandas numpy notebook```

## Notebook extension
```
conda install -c conda-forge jupyter_contrib_nbextensions
conda install ipyparallel

jupyter nbextension enable codefolding/main
jupyter nbextension enable toc2/main
jupyter nbextension enable collapsible_headings/main
```

### Install pytorch, pytorch-lightning, tensorflow

```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

conda install pyg -c pyg

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install tensorflow

pip install pytorch-lightning
```

### Install torch-geometric
```
export TORCH=1.11.0
export CUDA=cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-$\{TORCH\}+$\{CUDA\}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-$\{TORCH\}+$\{CUDA\}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-$\{TORCH\}+$\{CUDA\}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$\{TORCH\}+$\{CUDA\}.html
pip install torch-geometric
```

### Install scikit-learn
pip install scikit-learn

### Graph
conda install -c stellargraph stellargraph

### Others
```
pip install commentjson
pip install snap-stanford
```