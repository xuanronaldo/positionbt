from setuptools import find_packages, setup

setup(
    name="positionbt",
    version="v1.1.1",
    author="Xuan Ronaldo",
    author_email="xuanronaldo@outlook.com",
    description="PositionBT is a simple, fast, and customizable backtesting framework "
    + "that directly evaluates trading strategies through position data (ranging from -1 to 1).",
    url="https://github.com/xuanronaldo/positionbt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "positionbt.visualization": ["templates/*.html"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "polars>=1.13.0",
        "pandas>=2.0.0",
        "plotly>=5.18.0",
    ],
)
