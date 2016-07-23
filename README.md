#### Use OpenCV to create crosshatch-style drawings and videos.

<img src='/examples/obama.png?raw=true' width='400px' style='display:inline-block;'>)
<img src='/examples/goldengate.png?raw=true' width='400px' align="right" style='display:inline-block;'>)
<img src='/examples/matrix.gif?raw=true' width='400px' align="right"  style='display:inline-block;'>)
<img src='/examples/emily.gif?raw=true' width='400px' align="right"  style='display:inline-block;'>)

#### To use:
```
$ make
$ ./crosshatched sources/goldengate.jpg
$ ./crosshatched --laplacian sources/goldengate.jpg
$ ./crosshatched path/to/a/video.mp4
```

This script is optimized for images and videos >= 1080 pixels wide. You can also
get decent results out of smaller images by tweaking the constants although
bigger seems to be better in most cases.

#### The algorithm:
- Calculate the gradient of the image
- Draw short, connected bezier lines parallel and perpendicular to that gradient
- Generate the "edge gradient" from either the Laplacian or Canny (default) algorithms
- Using slightly tweaked rules, draw more bezier lines parallel to the edge gradient

#### Dependencies:
 - g++ (Part of the GNU Compiler Collection)
 - OpenCV

#### A note:
I'm generally a Python programmer so please forgive my shitty C++ code.
