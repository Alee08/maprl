#!/usr/bin/env python3
from setuptools import setup, find_packages
import multiagentplanning_rl

long_description = "Multi-Agent Planning Reinforcement Learning"

setup(
    name="multiagentplanning_rl",
    version="0.1.0",
    description="Multi-Agent Planning Reinforcement Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alessandro Trapasso",
    author_email="Ale.trapasso8@gmail.com",
    url="",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "multiagentplanning_rl.render.img": ["*"],
    },
    # Consigliato: specifica la versione minima di Python
    python_requires=">=3.10",
    install_requires=[
        "multiagent-rl-rm==0.1.1",
        "up-tamer",
        "up-enhsp",
        "imageio",
        "wandb",
        # opzionale ma utile se non già dipendenza transitiva:
        # "unified-planning>=1.0",
    ],
    extras_require={
        "data_analysis": [],
    },
    license="APACHE",
    keywords="learning multiagent rewardmachine reinforcementlearning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
    ],
)
