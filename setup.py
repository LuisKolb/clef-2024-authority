from setuptools import setup, find_packages

setup(
    name="clef-2024-checkthat-lkolb",
    version="0.1.0",
    description="Fact-Checking Claims using Authority Retrieval",
    author="Luis Kolb",
    author_email="kolb.luis@gmail.com",
    url="https://luiskolb.at",
    install_requires=[
        "pyserini",
        "jupyter",
        "faiss-cpu",
        "scikit-learn",
        "python-terrier",
        "ir-measures",
        "jsonlines"
    ],
    packages=find_packages(["clef", "clef2024-checkthat-lab/task5/*"]),
)
