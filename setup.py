from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ice_sheet_analysis',
    version='0.1.0',
    description='Multi-resolution ice sheet analysis tools',
    author='Pierre Thodoroff',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=3.6',
    install_requires=required,
)