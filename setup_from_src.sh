module purge
module load cuda cudnn cmake scipy-stack python/3.10

virtualenv --no-download ~/pyenv/degeneracy
source ~/pyenv/degeneracy/bin/activate

pip install --no-index --upgrade pip

# install torch from source
cd pytorch
git submodule sync
git submodule update --init --recursive

pip install -r requirements.txt

export _GLIBCXX_USE_CXX11_ABI=1
export USE_CUDA=1

python setup.py develop
