# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

pip install huggingface_hub
~/.local/bin/huggingface-cli login --token $HF_TOKEN

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_basic_correctness || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_basic_correctness \
       	rocm /bin/bash -c "export HF_TOKEN=$$HF_TOKEN; python3 -m pytest -v -s vllm/tests/basic_correctness"

