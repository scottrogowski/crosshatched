all: crosshatched.cpp
	g++ crosshatched.cpp -Ofast -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -o fast_gradient

crosshatched: crosshatched.cpp
	g++ crosshatched.cpp -Ofast -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -o fast_gradient
