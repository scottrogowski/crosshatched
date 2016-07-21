all: fast_gradient.cpp
	g++ fast_gradient.cpp -Ofast -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -o fast_gradient

fast_gradient: fast_gradient.cpp
	g++ fast_gradient.cpp -Ofast -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -o fast_gradient
