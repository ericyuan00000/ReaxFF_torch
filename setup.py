import setuptools
from os import path
import reaxff

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name=reaxff.__name__,
        version=reaxff.__version__,
        author='Eric Yuan',
        author_email='ericyuan@berkeley.edu',
        project_urls={
            'Source': 'https://github.com/ericyuan00000/Torch-ReaxFF',
        },
        description='PyTorch implementation of ReaxFF',
        long_description=long_description,
        long_description_content_type='text/markdown',
        keywords=[
            'Machine Learning', 
            'Data Mining', 
            'Quantum Chemistry',
            'Molecular Dynamics',
            ],
        license='MIT',
        packages=setuptools.find_packages(),
        install_requires=['torch'],
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            ],
        zip_safe=False,
    )
