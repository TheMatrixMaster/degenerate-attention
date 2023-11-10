module purge
module load scipy-stack python/3.10

virtualenv --no-download ~/pyenv/degeneracy
source ~/pyenv/degeneracy/bin/activate

pip install --no-index --upgrade pip

# Install torch from dist
pip install --no-index -r requirements.txt

# Copy src files to support degenerate attention
torch_path=$(python -c "import torch; print(torch.__path__[0])")

cp deg-attn-src/functional.py $torch_path/nn/functional.py
cp deg-attn-src/activation.py $torch_path/nn/modules/activation.py
cp deg-attn-src/transformer.py $torch_path/nn/modules/transformer.py

echo "Done!"