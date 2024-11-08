FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /TopoMamba

COPY . .

RUN pip install --upgrade pip
RUN chmod +x .
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu115.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu115.html
RUN pip install torch_geometric
RUN pip install pre-commit jupyterlab notebook ipykernel pandas ipdb

RUN pip install causal-conv1d==1.2.0.post2
RUN pip install mamba-ssm
RUN pip install pytorch-lightning
RUN pip install wandb

RUN pip install topomodelx
RUN apt update
RUN apt-get install -y git
RUN chown -R root /TopoMamba
