# Build docker container with HPCCM
# The use of sed is to fix a broken link
FILEBASE="horovod-torch-tensorboard"
D_ID=$(hpccm --recipe $FILEBASE.py --format docker \
| sed 's_http://ftpmirror\.gnu.org/[a-z\./7\-]\+_http://ftp.acc.umu.se/mirror/gnu.org/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.xz_' \
| sed 's_var/tmp/gcc-7_var/tmp/gcc-7.5.0_g' \
| sudo docker build - | tee /dev/stderr | sed -n 's/^Successfully built \([0-9a-f]\+\)/\1/gp')


if [ -z "$D_ID" ]; then

    echo Could not build singularity. Docker ID not found.

else

    # Docker -> Tarball
    sudo docker save $D_ID -o $FILEBASE.tar &&

    # Tarball -> Singularity
    singularity build --sandbox $FILEBASE $FILEBASE.tar

fi

