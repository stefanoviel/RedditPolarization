# RedditPolarization

module load stack/2024-05  gcc/13.2.0 python/3.11.6_cuda

postgresql/13.2 ??

To install cuml please refer to https://docs.rapids.ai/install

this needs to be done first: 

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* cuml-cu12==24.6.*

then: 
pip install -r requirements.txt
