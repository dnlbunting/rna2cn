from setuptools import setup, find_packages


setup(
    name = "rna2cn",
    version = '0.0.0',
    author = "Daniel Bunting",
    author_email = "daniel.bunting@crick.ac.uk",
    url = "https://github.com/dnlbunting/rna2cn/",
    packages=['rna2cn'],
    entry_points={
        'console_scripts': [
            'rna2cn = rna2cn.rna2cn_run:main',
            ]
        }
)