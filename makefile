all: crosshatched.cpp
	g++ crosshatched.cpp -Wc++11-extensions -Ofast -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -o crosshatched

crosshatched: crosshatched.cpp
	g++ crosshatched.cpp -Wc++11-extensions -Ofast -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -o crosshatched
