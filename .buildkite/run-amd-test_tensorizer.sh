# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_tensorizer || true; }
trap remove_docker_container EXIT
remove_docker_container

apt-get install curl libsodium23 && pytest -v -s tensorizer_loader

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host \
	--name rocm_test_tensorizer rocm /bin/bash -c "apt install curl libsodium23; \
                python3 -m pytest -v -s vllm/tests/tensorizer_loader"

