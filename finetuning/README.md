## Setup

```
conda create --name plue_ft python==3.8
conda activate plue_ft
pip install -r requirements.txt

# for fp16 support
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
```