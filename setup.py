from setuptools import setup
import re
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    version=find_version("dpk", "__init__.py"),
    name="dpk-forecast",
    description="Make long-term probabilistic forecasts using the Deep Probabilistic Koopman framework",
    author="Alex Mallen; built on code from Henning Lange",
    author_email="atmallen@uw.edu",
    url="https://github.com/AlexTMallen/dpk",
    packages=["dpk"],
    install_requires=required,
    setup_requires=[],
)