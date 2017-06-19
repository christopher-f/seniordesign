# Had lots of help from here:
# https://www.scivision.co/anaconda-python-opencv3/#prereqs-linuxwindows-subsystem-for-linux

# Enable CUDA since I have a Nvidia GPU, disable if you dont
# Refer to this link: https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-OpenCV-3.1-Installation-Guide
# Necessary workaround for a Nvidia bug with CUDA 7.5, bug is fixed in CUDA 8.0
# Until then, keep using the following workaround: -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" 

# DPYTHON_LIBRARIES: 
# Got this info from:http://stackoverflow.com/questions/20582270/distribution-independent-libpython-path

# How to get the path of any Python module:
# https://leemendelowitz.github.io/blog/how-does-python-find-packages.html
# This was used to get the path for your numpy install.
# If imp.find_module()'s API ever changes, this will break - I expect index 1
# of the returned tuple to contain the numpy's install path.
# [Set OFF if you don't have NVIDIA GPU:]
# -DWITH_CUDA=OFF \
# -DWITH_CUBLAS=OFF \
# [Delete if you don't have NVIDIA GPU:]
# Delete -DCUDA_NVCC_FLAGS
cmake -DBUILD_TIFF=ON \
-DBUILD_opencv_java=OFF \
-DWITH_CUDA=ON \
-DWITH_OPENGL=ON \
-DWITH_OPENCL=ON \
-DWITH_IPP=ON \
-DWITH_TBB=ON \
-DWITH_EIGEN=ON \
-DWITH_V4L=ON \
-DWITH_VTK=OFF \
-DWITH_QT=ON \
-DWITH_CUBLAS=ON \
-DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-DPYTHON3_EXECUTABLE=$(which python3) \
-DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON3_LIBRARIES=$(python3 -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import imp; print(imp.find_module('numpy')[1])") \
-DPYTHON_DEFAULT_EXECUTABLE=$(which python3) ..
