#! /usr/bin/env python
import os
from setuptools import setup, find_packages

# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DISTNAME = 'nmrnn'
DESCRIPTION = 'Neuromodulated RNN models.'
with open('README.md') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Julia Costacurta'
MAINTAINER_EMAIL = 'jcostac@stanford.edu'
URL = ''
LICENSE = 'MIT'
DOWNLOAD_URL = ''
VERSION = '0.0.1'


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.9',
          ],
          packages=find_packages(),
          )
