#### Use OpenCV to create crosshatch-style drawings and videos.

<img src='/examples/obama.png?raw=true' width='390px' style='display:inline-block;'>
<img src='/examples/goldengate.png?raw=true' width='390px' style='display:inline-block;'>
<img src='/examples/thailand.png?raw=true' width='390px' style='display:inline-block;'>
<img src='/examples/matrix.gif?raw=true' width='390px' style='display:inline-block;'>
<img src='/examples/emily.gif?raw=true' width='390px' style='display:inline-block;'>

#### Try out a canvas-based implementation at https://www.sketchify.me

#### To use:
```
$ make
$ ./crosshatched sources/goldengate.jpg
$ ./crosshatched --laplacian sources/thailand.jpg
$ ./crosshatched path/to/a/video.mp4
```

This script is optimized for images and videos >= 1080 pixels wide. You can also
get decent results out of smaller images by tweaking the constants although
bigger seems to be better in most cases.

There are a bunch more flags too worth playing around with

#### The algorithm:
- Calculate the gradient of the image
- Draw short, connected bezier lines parallel and perpendicular to that gradient
- Generate the "edge gradient" from either the Laplacian or Canny (default) algorithms
- Using slightly tweaked rules, draw more bezier lines parallel to the edge gradient

#### Dependencies:
 - g++ (Part of the GNU Compiler Collection)
 - OpenCV
 - Tons on other stuff (install everything one error at a time)

#### TODO:
I'd love to pipe this through a neural-network style-transfer first to get something really artistic

#### A note:
I'm generally a Python programmer so please forgive my shitty C++ code. I'm also not at work so I'm not keeping my code clean

