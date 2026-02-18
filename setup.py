from setuptools import setup, find_packages

setup(
      name='proteinfoundation',
      version='1.0.0',
      description='proteina',
      packages=find_packages(),
      package_dir={
          'proteinfoundation': './proteinfoundation',
          'openfold': './openfold',
          'graphein_utils': './graphein_utils'
      },
      install_requires=[
          "jax[cpu]>=0.4.20",
          "jaxtyping",
          "torch>=2.1.0",
          "torch-geometric>=2.6.1",
          "torch-sparse",
          "torch-scatter",
          "torch-cluster",
          "transformers>=4.48.3",
          "lightning>=2.0",
          "hydra-core>=1.3",
          "numpy>=1.23.5,<2.0",
          "pandas>=1.5.3",
          "biopandas>=0.5.1",
          "wandb>=0.17.0",
          "tqdm>=4.66.4",
          "einops>=0.6",
          "dm-tree>=0.1.8",
          "loguru>=0.7.2",
          "python-dotenv>=1.0.1",
          "wget>=3.2",
          "ml-collections>=0.1.1",
          "cpdb-protein",
          "loralib",
          "biopython",
          "biotite>=1.0,<2.0",
          # NOTE: mmseqs2 is bioconda-only — install separately
          # e.g. via: conda install -c bioconda mmseqs2, or an HPC module
      ],
)