# setup.py
"""
Setup script for easy installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qcgan-eeg-seizure",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Quantum Conditional GAN for EEG Seizure Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum_eeg_seizure_detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qcgan-download=download_data:main",
            "qcgan-preprocess=preprocess_data:main",
            "qcgan-train=train:main",
            "qcgan-evaluate=evaluate:main",
            "qcgan-demo=demo:main",
        ],
    },
)