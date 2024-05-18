from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
        requirements = []
        for line in lines:
            line = line.strip()
            # Ignore comments and empty lines
            if line and not line.startswith('#'):
                requirements.append(line)
        return requirements

install_requires = parse_requirements('requirements.txt')

setup(
    name='LLM-Interrogation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'lint=lint.entrypoint:main',
        ],
    },
    author='Zhuo Zhang',
    author_email='research@zzhang.xyz',
    # description='LLM Interrogation',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='https://github.com/yourusername/your_project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
