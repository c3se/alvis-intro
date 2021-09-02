Stage0 += baseimage(image='nvidia/cuda:10.2-devel-centos7')

# OpenMPI
compiler = gnu(
    extra_repository=True,
    version="7",
    source=True,
)
Stage0 += ofed()
Stage0 += compiler
Stage0 += openmpi(
    cuda=True,
    infiniband=True,
    toolchain=compiler.toolchain,
    version='4.0.5',
)

# Install Python
pyversion = "3.9.6"
Stage0 += yum(ospackages=["openssl-devel", "libffi-devel", "zlib-devel"])
Stage0 += generic_build(
    url=f"https://www.python.org/ftp/python/{pyversion}/Python-{pyversion}.tgz",
    unpack=True,
    build=["./configure --enable-optimizations"],
    install=["make install", "python --version", "python3 --version"],
    toolchain=compiler.toolchain,
)
#Stage0 += python(python2=False, python3=True, devel=True)

# Install relevant python packages
# n.b. these are not compiled for Alvis and might be lacking in performance
Stage0 += nccl()
Stage0 += cmake(toolchain=compiler.toolchain, eula=True)
Stage0 += environment(variables=dict(
    CXX="/usr/local/gnu/bin/g++",
    HOROVOD_GPU_Operations="NCCL",
    HOROVOD_WITH_PYTORCH="1",
    HOROVOD_WITHOUT_TENSORFLOW="1",
    HOROVOD_WITHOUT_MXNET="1",
))
Stage0 += pip(packages=["wheel", "torch", "tensorboard", "horovod[pytorch]"], pip="pip3")
