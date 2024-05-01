# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

for((i=0;i<`rocm-smi -i | grep "Device ID" | wc -l`;i++)); do 
    #rocm-smi -gpureset -d $i; 
done

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_models || true; \
			docker rm -f rocm_test_oot_registration || true; \
			docker rm -f rocm_test_models_py || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_models \
	-e HF_TOKEN rocm /bin/bash -c "cd vllm/tests; /bin/bash ../.buildkite/download-images.sh; \
	 python3 -m pytest -v -s models --ignore=models/test_llava.py --ignore=models/test_mistral.py"

docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_oot_registration \
	        rocm python3 -m pytest -v -s vllm/tests/models/test_oot_registration.py

docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_models_py \
	                -e HF_TOKEN rocm python3 -m pytest -v -s vllm/tests/models/test_models.py

