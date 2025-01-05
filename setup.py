from setuptools import setup, find_packages

setup(
    name='aimindbot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'google-generativeai',
        'colorama',
       'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'aimindbot = main:main',
        ],
    },
)
