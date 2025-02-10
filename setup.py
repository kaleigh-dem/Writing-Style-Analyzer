from setuptools import setup, find_packages

setup(
    name="writing_style_analyzer",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit",
        "torch",
        "transformers",
        "openai",
        "pandas"
    ],
    entry_points={
        "console_scripts": [
            "run-analyzer=app:main",
        ],
    },
    author="Kaleigh DeMartino",
    author_email="kaleigh.demartino@gmail.com",
    description="An AI-powered tool for analyzing and generating writing styles.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourgithub/Writing_Style_Analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)