import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gedml",
    version="0.1.4",
    author="Borui Zhang",
    author_email="zhang-br21@mails.tsinghua.edu.cn",
    description="GeDML is an easy-to-use generalized deep metric learning library, \
        which contains state-of-the-art deep metric learning algorithms and \
        auxiliary modules to build end-to-end compute vision systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zbr17/GeDML",
    project_urls={
        "Docs": "https://zbr17.github.io/GeDML/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "torch >= 1.7.0",
        "faiss-gpu == 1.5.3",
        "torchvision >= 0.8.0",
        "pretrainedmodels >= 0.7.4",
        "libtmux >= 0.8.5",
        "numpy >= 1.19.2",
        "tqdm >= 4.59.0",
        "pandas >= 1.1.3",
        "scipy >= 1.5.4",
        "scikit-learn >= 0.24.1",
        "tensorboard >= 2.4.0",
        "wandb >= 0.10.22",
        "timm == 0.3.2",
        "graphviz",
        "torchdistlog"
    ]
)