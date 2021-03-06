import errno
import os
import cv2

class Video:
    """ Object containing video metadata. """

    def __init__(self, cls, name, path):
        """
        Instantiate a 'Video' object.

        :param cls: The video's class - name of person
        :type cls: str
        :param name: The video's name
        :type name: str
        :param path: Path to video on disk
        :type path: str
        """
        assert cls is not None
        assert name is not None
        assert path is not None

        self.cls = cls
        self.name = name
        self.path = path

    def getCap(self):
        """ Get the video capture in cv2.VideoCapture form """
        return cv2.VideoCapture(self.path)

    def __repr__(self):
        """ String representation when printed """
        return "({} {})".format(self.cls, self.name)


def mkdirP(path):
    """
    Create a directory and don't error if the path exists already.

    If directory exists, do nothing
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def iterVids(directory):
    """
    Iterate through videos in a directory.

    Videos should be organized in subdirectories named by who is in the video
    """
    assert directory is not None

    exts = [".mp4", ".mov", ".webm"]  # Accept more later

    for subdir, dirs, files in os.walk(directory):
        for path in files:
            (vidClass, fName) = (os.path.basename(subdir), path)
            (vidName, ext) = os.path.splitext(fName)
            if ext.lower() in exts:
                yield Video(vidClass, vidName, os.path.join(subdir, fName))
