from setuptools import setup
from setuptools import find_packages

# load README file
with open(file="README.md", mode='r') as readme_handle:
    long_description = readme_handle.read()

setup(
        name='cy_im_utils',
        author='M. Cyrus Daugherty',
        author_email='michael.daugherty@nist.gov',
        #    major. minor. maintenance
        version='0.0.1',
        description="Image processing utils for tomography",
        long_description=long_description,
        long_description_content_type='text/markdown',
        install_requires=[
            "numpy>=1.21.5",
            "cupy>=10.1.0",
            "pillow>=8.4.0",
            "matplotlib>=3.3.4",
            "ipywidgets>=7.6.5",
            "scipy>=1.7.3",
            "tqdm>=4.60.0",
            "numba>=0.55.1",
            "astropy>=4.2.1",
            ],
        keywords='tomography',
        packages=find_packages(where="cy_im_utils")
        )
