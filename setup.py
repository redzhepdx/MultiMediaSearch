from setuptools import find_namespace_packages, setup

target_python_version = '>=3.8.0'

setup(
    name='multimedia_search',
    version='0.0.1',
    description="Multimedia search service",
    python_requires=target_python_version,
    packages=find_namespace_packages(include=['multimedia_search', 'multimedia_search.*']),
    include_package_data=True,
)
