module purge
module load gcc arrow cuda scipy-stack python/3.10

virtualenv --no-download ~/pyenv/degeneracy
source ~/pyenv/degeneracy/bin/activate

pip install --no-index --upgrade pip

# Install packages including jupyter-lab
pip install --no-index -r requirements.txt

# Add wrapper script to launch jupyter lab
echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter lab --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/jupyterlab.sh
chmod u+x $VIRTUAL_ENV/bin/jupyterlab.sh

# Copy src files to support degenerate attention
torch_path=$(python -c "import torch; print(torch.__path__[0])")
transformers_path=$(python -c "import transformers; print(transformers.__path__[0])")

cp deg-attn-src/functional.py $torch_path/nn/functional.py
cp deg-attn-src/activation.py $torch_path/nn/modules/activation.py
cp deg-attn-src/transformer.py $torch_path/nn/modules/transformer.py
cp deg-attn-src/configuration_gpt2.py $transformers_path/models/gpt2/configuration_gpt2.py
cp deg-attn-src/modeling_gpt2.py $transformers_path/models/gpt2/modeling_gpt2.py

echo "Done!"