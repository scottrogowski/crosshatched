mac: crosshatched.cpp
	g++ crosshatched.cpp -o crosshatched -Ofast -lm -I/usr/local/Cellar/opencv3/HEAD-44e5d26_4/include/opencv -I/usr/local/Cellar/opencv3/HEAD-44e5d26_4/include -L/usr/local/Cellar/opencv3/HEAD-44e5d26_4/lib  -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core -std=c++11

ubuntu: crosshatched.cpp
	g++ crosshatched.cpp -o crosshatched -Ofast -lm `pkg-config opencv --cflags --libs` -std=c++11
