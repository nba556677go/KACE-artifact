#install docker 
echo "install docker..."
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

#nvidia container
echo "install nvidia container toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

#configure docker engine
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo groupadd docker
sudo usermod -aG docker $USER
sudo docker pull nba556677/ml_tasks:latest
#add nightsight compute requirments
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
sudo sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'

#add $1==profile
if [ "$1" == "profile" ]
then
    echo "install nvidia ncu tool for profiling..."
    #install nvidia tools
    #setup nsys
    #check version - https://developer.nvidia.com/nsight-systems/get-started#platforms
    #wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_2/nsightsystems-linux-public-2024.2.1.106-3403790.run
    #chmod +x nsightsystems-linux-public-2024.2.1.106-3403790.run
    #./nsightsystems-linux-public-2024.2.1.106-3403790.run

    #setup nsight=compute
    #might need manually download and copy run file to server - https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2024_1_1/nsight-compute-linux-2024.1.1.4-33998838.run
    #another source - https://developer.nvidia.com/cuda-12-3-1-download-archive
    #wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2024_1_1/nsight-compute-linux-2024.1.1.4-33998838.run
    #./nsight-compute-linux-2024.1.1.4-33998838.run
    #/usr/local/cuda-12.3/

    wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
    sudo sh cuda_12.3.0_545.23.06_linux.run

    #set install path = /home/cc/ncu-2023.3.0
fi


