
# create a new environment with Python 3.10
conda create -n yolo310 python=3.10 -y
conda activate yolo310

pip install --upgrade pip

# Reinstall your packages
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg
pip install streamlit==1.28.2 opencv-python-headless==4.11.0.86 pillow==9.5.0 numpy==1.26.4


pip install streamlit ultralytics opencv-python-headless numpy pillow
pip install torch torchvision




MASK/
├── images/               ← All images
├── annotations/          ← All XML files
├── labels/               ← YOLO-format .txt files (generated)
├── datasets/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
├── data.yaml             ← Configuration file for training


Package                   Version
------------------------- ----------------
absl-py                   2.3.1
aioice                    0.10.1
aiortc                    1.13.0
altair                    5.5.0
annotated-types           0.7.0
anyio                     4.9.0
apturl                    0.5.2
argon2-cffi               25.1.0
argon2-cffi-bindings      21.2.0
arrow                     1.3.0
asttokens                 3.0.0
astunparse                1.6.3
async-lru                 2.0.5
attrs                     25.3.0
av                        14.4.0
babel                     2.17.0
bcrypt                    3.2.0
beautifulsoup4            4.13.4
bleach                    6.2.0
blinker                   1.4
blis                      1.3.0
Brlapi                    0.8.3
cachetools                5.5.2
catalogue                 2.0.10
certifi                   2020.6.20
cffi                      1.17.1
chardet                   4.0.0
charset-normalizer        3.4.2
click                     8.0.3
cloudpathlib              0.21.1
colorama                  0.4.4
comm                      0.2.3
command-not-found         0.3
confection                0.1.5
contourpy                 1.3.2
cryptography              45.0.7
cupshelpers               1.0
cycler                    0.12.1
cymem                     2.0.11
dbus-python               1.2.18
debugpy                   1.8.15
decorator                 5.2.1
defer                     1.0.6
defusedxml                0.7.1
distro                    1.7.0
distro-info               1.1+ubuntu0.2
dnspython                 2.8.0
duplicity                 0.8.21
en_core_web_sm            3.8.0
et_xmlfile                2.0.0
exceptiongroup            1.3.0
executing                 2.2.0
fasteners                 0.14.1
fastjsonschema            2.21.1
filelock                  3.19.1
flatbuffers               25.2.10
fonttools                 4.59.0
fqdn                      1.5.1
fsspec                    2025.9.0
future                    0.18.2
gast                      0.6.0
gitdb                     4.0.12
GitPython                 3.1.45
google-crc32c             1.7.1
google-pasta              0.2.0
graphviz                  0.21
grpcio                    1.74.0
h11                       0.16.0
h5py                      3.14.0
html5lib                  1.1
httpcore                  1.0.9
httplib2                  0.20.2
httpx                     0.28.1
idna                      3.3
ifaddr                    0.2.0
importlib-metadata        4.6.4
ipykernel                 6.30.0
ipython                   8.37.0
isoduration               20.11.0
jedi                      0.19.2
jeepney                   0.7.1
Jinja2                    3.1.6
joblib                    1.5.1
json5                     0.12.0
jsonpointer               3.0.0
jsonschema                4.25.0
jsonschema-specifications 2025.4.1
jupyter_client            8.6.3
jupyter_core              5.8.1
jupyter-events            0.12.0
jupyter-lsp               2.2.6
jupyter_server            2.16.0
jupyter_server_terminals  0.5.3
jupyterlab                4.4.5
jupyterlab_pygments       0.3.0
jupyterlab_server         2.27.3
keras                     3.11.2
keyring                   23.5.0
kiwisolver                1.4.8
langcodes                 3.5.0
language_data             1.3.0
language-selector         0.1
lark                      1.2.2
launchpadlib              1.10.16
lazr.restfulclient        0.14.4
lazr.uri                  1.0.6
libclang                  18.1.1
lockfile                  0.12.2
louis                     3.20.0
lxml                      4.8.0
macaroonbakery            1.3.1
Mako                      1.1.3
marisa-trie               1.2.1
Markdown                  3.8.2
markdown-it-py            3.0.0
MarkupSafe                3.0.2
matplotlib                3.10.3
matplotlib-inline         0.1.7
mdurl                     0.1.2
mistune                   3.1.3
ml_dtypes                 0.5.3
monotonic                 1.6
more-itertools            8.10.0
mpmath                    1.3.0
murmurhash                1.0.13
namex                     0.1.0
narwhals                  2.4.0
nbclient                  0.10.2
nbconvert                 7.16.6
nbformat                  5.10.4
nest-asyncio              1.6.0
netifaces                 0.11.0
networkx                  3.4.2
notebook_shim             0.2.4
numpy                     1.26.4
nvidia-cublas-cu12        12.8.4.1
nvidia-cuda-cupti-cu12    12.8.90
nvidia-cuda-nvrtc-cu12    12.8.93
nvidia-cuda-runtime-cu12  12.8.90
nvidia-cudnn-cu12         9.10.2.21
nvidia-cufft-cu12         11.3.3.83
nvidia-cufile-cu12        1.13.1.3
nvidia-curand-cu12        10.3.9.90
nvidia-cusolver-cu12      11.7.3.90
nvidia-cusparse-cu12      12.5.8.93
nvidia-cusparselt-cu12    0.7.1
nvidia-nccl-cu12          2.27.3
nvidia-nvjitlink-cu12     12.8.93
nvidia-nvtx-cu12          12.8.90
nvidia-pyindex            1.0.9
oauthlib                  3.2.0
olefile                   0.46
opencv-python             4.12.0.88
opencv-python-headless    4.11.0.86
openpyxl                  3.1.5
opt_einsum                3.4.0
optree                    0.17.0
overrides                 7.7.0
packaging                 23.2
pandas                    2.3.1
pandocfilters             1.5.1
paramiko                  2.9.3
parso                     0.8.4
pexpect                   4.8.0
Pillow                    9.5.0
pip                       25.2
platformdirs              4.3.8
polars                    1.33.1
preshed                   3.0.10
prometheus_client         0.22.1
prompt_toolkit            3.0.51
protobuf                  4.25.8
psutil                    7.0.0
ptyprocess                0.7.0
pure_eval                 0.2.3
py-cpuinfo                9.0.0
pyarrow                   21.0.0
pycairo                   1.20.1
pycparser                 2.22
pycups                    2.0.1
pydantic                  2.11.7
pydantic_core             2.33.2
pydeck                    0.9.1
pyee                      13.0.0
pygame                    2.6.1
Pygments                  2.19.2
PyGObject                 3.42.1
PyJWT                     2.3.0
pylibsrtp                 0.12.0
pymacaroons               0.13.0
PyNaCl                    1.5.0
pyOpenSSL                 25.1.0
pyparsing                 2.4.7
pyRFC3339                 1.1
python-apt                2.4.0+ubuntu4
python-dateutil           2.9.0.post0
python-debian             0.1.43+ubuntu1.1
python-json-logger        3.3.0
pytz                      2022.1
pyxdg                     0.27
PyYAML                    5.4.1
pyzmq                     27.0.0
referencing               0.36.2
reportlab                 3.6.8
requests                  2.32.4
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rfc3987-syntax            1.1.0
rich                      13.9.4
rpds-py                   0.26.0
scikit-learn              1.7.1
scipy                     1.13.1
scour                     0.38.2
screen-resolution-extra   0.0.0
seaborn                   0.13.2
SecretStorage             3.3.1
Send2Trash                1.8.3
setuptools                80.9.0
shellingham               1.5.4
six                       1.16.0
smart_open                7.3.0.post1
smmap                     5.0.2
sniffio                   1.3.1
soupsieve                 2.7
spacy                     3.8.7
spacy-legacy              3.0.12
spacy-loggers             1.0.5
srsly                     2.5.1
ssh-import-id             5.11
stack-data                0.6.3
streamlit                 1.28.2
streamlit-webrtc          0.63.4
sympy                     1.14.0
systemd-python            234
tenacity                  8.5.0
tensorboard               2.20.0
tensorboard-data-server   0.7.2
tensorflow                2.20.0
termcolor                 3.1.0
terminado                 0.18.1
thinc                     8.3.6
threadpoolctl             3.6.0
tinycss2                  1.4.0
toml                      0.10.2
tomli                     2.2.1
torch                     2.2.0+cpu
torchvision               0.17.0+cpu
tornado                   6.5.1
tqdm                      4.67.1
traitlets                 5.14.3
triton                    3.4.0
typer                     0.16.0
types-python-dateutil     2.9.0.20250708
typing_extensions         4.14.1
typing-inspection         0.4.1
tzdata                    2025.2
tzlocal                   5.3.1
ubuntu-drivers-common     0.0.0
ubuntu-pro-client         8001
ufw                       0.36.1
ultralytics-thop          2.0.17
unattended-upgrades       0.1
uri-template              1.3.0
urllib3                   1.26.5
usb-creator               0.3.7
validators                0.35.0
wadllib                   1.3.6
wasabi                    1.1.3
watchdog                  6.0.0
wcwidth                   0.2.13
weasel                    0.4.1
webcolors                 24.11.1
webencodings              0.5.1
websocket-client          1.8.0
Werkzeug                  3.1.3
wheel                     0.45.1
wrapt                     1.17.2
xdg                       5
xgboost                   3.0.2
xkit                      0.0.0
zipp                      1.0.0


