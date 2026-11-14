"""Package definition for carma_vla."""

from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "carma_vla"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Erick",
    description="Live VLA pipeline: stereo cameras and operator voice into the policy server.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"vla_node = {PACKAGE_NAME}.vla_node:main",
        ],
    },
)
