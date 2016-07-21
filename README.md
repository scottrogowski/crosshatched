#### Create crosshatch-style drawings and videos.

![Alt text](/examples/obama.png?raw=true =400x)
![Alt text](/examples/goldengate.png?raw=true =400x)

#### To use
$ make
$ ./crosshatched sources/goldengate.jpg
$ ./crosshatched --laplacian sources/goldengate.jpg
$ ./crosshatched path/to/a/video.mp4

The larger the image, the better this tends to work. Smaller images can be done
by tweaking the constants. Lowering the density and the line_length in particular
would get you most of the way there.

#### The algorithm:
- Calculate the gradient of the image
- Draw short, connected bezier lines parallel and perpendicular to that gradient
- Generate the "edge gradient" from either the Laplacian or Canny (default) algorithms
- Using slightly tweaked rules, draw more bezier lines parallel to the edge gradient

#### Dependencies:
 - g++ (Part of the GNU Compiler Collection)
 - OpenCV
