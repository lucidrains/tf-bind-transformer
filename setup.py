from setuptools import setup, find_packages

setup(
  name = 'tf-bind-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.31',
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
    'click',
    'einops>=0.3',
    'enformer-pytorch>=0.3.1',
    'fair-esm',
    'logavgexp-pytorch',
    'polars',
    'torch>=1.6',
    'transformers>=4.0',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
