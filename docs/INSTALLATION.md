# Installation and Setup

This guide covers all the steps needed to install and configure TuRTLe for API-based inference and local Docker evaluation.

- For cluster/HPC setup with vLLM and Singularity, please see [LOCAL_INFERENCE.md](../LOCAL_INFERENCE.md)

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv)
- Docker (for local API evaluation only)

## Installing uv

On macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or using pip:

```bash
pip install uv
```

## Clone the Repository

```bash
git clone --recursive https://github.com/HPAI-BSC/TuRTLe.git
cd TuRTLe
```

## Install Dependencies

Initialize the project and install dependencies:

```bash
uv init
uv add -r requirements.txt
```

This will create a virtual environment and install all required packages from `requirements.txt`.

## Docker Setup (for Local Evaluation)

Docker CE (Community Edition) with a recent version is required for running local evaluations with EDA tools.

### Install Docker

Install Docker CE from https://docs.docker.com/get-docker/

### Configure Docker Permissions

Add your user to the docker group to run Docker without sudo permissions:

```bash
# Add current user to docker group
sudo usermod -aG docker $USER

# Verify Docker works without sudo
docker --version
```

### Pull the Evaluation Image

Pull the TuRTLe evaluation Docker image that contains all EDA tools:

```bash
docker pull ggcr0/turtle-eval:2.3.4
```

## Next Steps

- See the main [README.md](../README.md) for quick start API inference and evaluation
- For cluster/HPC setup with vLLM and Singularity, please see [LOCAL_INFERENCE.md](../LOCAL_INFERENCE.md)
