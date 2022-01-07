from setuptools import setup, find_packages

setup(
  name = 'tf-bind-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Transformer for Transcription Factor Binding',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/tf-bind-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism',
    'transformers',
    'transcription factors',
    'gene expression'
  ],
  install_requires=[
    'biopython',
    'einops>=0.3',
    'enformer-pytorch>=0.1.19',
    'fair-esm',
    'pandas',
    'torch>=1.6',
    'transformers>=4.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
