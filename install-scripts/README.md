# Installation Instructions for OpenCV 3.1
- Make sure you're in your Anaconda environment. Run the following commands to get Qt5/PyQt5, or else OpenCV will crash.
```bash
conda update qt
conda update pyqt
```
- Download [OpenCV 3.1 Release](http://opencv.org/downloads.html)

```bash
unzip -a opencv-3-1.0.zip
cd opencv*
mkdir build
cd build
``` 
- Disable any Nvidia-specific options for compilation in `cmake_opencv3.sh` if you don't have a discrete GPU.
- Move the CMake install script into the build folder.
```bash
chmod +x cmake_opencv3.sh
./cmake_opencv3.sh
make -j4
make install
```
Then you're good! You should now be able to run the example application :)
