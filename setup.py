# coding: UTF-8
from setuptools import setup


from setuptools.extension import Extension

# TODO:
# - Wrap learning.
# - Make LabelCompatibility, UnaryEnergy, PairwisePotential extensible? (Maybe overkill?)


# If Cython is available, build using Cython.
# Otherwise, use the pre-built (by someone who has Cython, i.e. me) wrapper `.cpp` files.
try:
    from Cython.Build import cythonize
    ext_modules = cythonize([
        Extension('pydensecrf.eigen', ['pydensecrf/eigen.pyx'], extra_compile_args=["-O0", "-g"]),
        Extension('pydensecrf.densecrf', ['pydensecrf/densecrf.pyx'], extra_compile_args=["-O0", "-g"]),
        Extension('pydensecrf.pairwise', ['pydensecrf/pairwise.pyx'], extra_compile_args=["-O0", "-g"]),
        Extension('pydensecrf.lattice', ['pydensecrf/lattice.pyx'], extra_compile_args=["-O0", "-g"])
        ])
except ImportError:
    from setuptools.extension import Extension
    ext_modules = [
        Extension("pydensecrf/eigen", ["pydensecrf/eigen.cpp", "pydensecrf/eigen_impl.cpp"], language="c++", include_dirs=["pydensecrf/densecrf/include"]),
        Extension("pydensecrf/densecrf", ["pydensecrf/densecrf.cpp", "pydensecrf/densecrf/src/densecrf.cpp", "pydensecrf/densecrf/src/unary.cpp", "pydensecrf/densecrf/src/pairwise.cpp", "pydensecrf/densecrf/src/permutohedral.cpp", "pydensecrf/densecrf/src/optimization.cpp", "pydensecrf/densecrf/src/objective.cpp", "pydensecrf/densecrf/src/labelcompatibility.cpp", "pydensecrf/densecrf/src/util.cpp", "pydensecrf/densecrf/external/liblbfgs/lib/lbfgs.c"], language="c++", include_dirs=["pydensecrf/densecrf/include", "pydensecrf/densecrf/external/liblbfgs/include"]),
    ]

setup(
    name="pydensecrf",
    version="1.0rc2",
    description="A python interface to Philipp Krähenbühl's fully-connected (dense) CRF code.",
    long_description="See the README.md at http://github.com/lucasb-eyer/pydensecrf",
    author="Lucas Beyer",
    author_email="lucasb.eyer.be@gmail.com",
    url="http://github.com/lucasb-eyer/pydensecrf",
    ext_modules=ext_modules,
    packages=["pydensecrf", "pydensecrf.tests"],
    setup_requires=['cython>=0.22'],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
