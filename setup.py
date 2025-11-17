from setuptools import setup
setup(
    name="findingdory",
    packages=["findingdory"],
    version="0.1",
    install_requires=[
        "pandas",
        "rtree",
        "wandb",
        "ipython",
        "ipdb",
        "json5",
    ],
    extras_require={
        "vlm_baseline": [
            "google-generativeai==0.8.3",
            "protobuf==3.20.2",
            "qwen-vl-utils",
            "openai",
            "transformers",
            "accelerate",
            "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git",
        ],
        "mapping_baseline": [
            "torch_geometric",
            "open3d",
            "scikit-image",
            "sophuspy",
            "scikit-fmm",
        ],
    },
)
