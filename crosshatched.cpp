/*
This will take an input image and stylize it to make it look like a crosshatch
drawing.

Opencv is the big requirement here and you will need to install that.
*/

#include <vector>
#include <queue>
#include <exception>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// #define DEBUG

#define BLACK 0
#define WHITE 255
#define PI 3.1415926
#define OUT_IMAGE_FILENAME "out.png"
#define OUT_VIDEO_FILENAME "out.mp4"
#define WINDOW_NAME "window"

int DENSITY = 5;
int SOBEL_KERNEL = 3;
int BLUR_SIZE = 20;
int BLOCK_SIZE_THRESHOLD = 35;
int LINE_LEN = 5;
int CANNY_LOWER = 80;
int CANNY_UPPER = 250;
int LAPLACE = 7;
int CANNY_GAUSS_BLUR = 1;
int PERPENDICULAR_TOLERANCE = 15;
int GRADIENT_TOLERANCE = 30;
int CANNY_TOLERANCE = 50;

int GRAD_LIMIT = 4;
int EDGE_LIMIT = 8;
int CANNY_DIV = 3;
int WHITE_THRESHOLD = 200;
int PERP_LESS_THRESHOLD = 50;




int get_int(Mat img, Point p) {
    return img.at<uchar>(p.y, p.x);
}

float get_fl(Mat img, Point p) {
    return img.at<float>(p.y, p.x);
}


float vect_len(Point pnt) {
    return sqrt((float)(pnt.x*pnt.x + pnt.y*pnt.y));
}

float dot(Point a, Point b) {
    // dot product
    return a.x*b.x + a.y*b.y;
}

float rads_btwn_vectors(Point a, Point b) {
    // shortest distance angle between two vectors (so max == PI)
    return acos(dot(a, b)/(vect_len(a)*vect_len(b)));
}

Mat zeroes_int(Mat img) {
    return img.zeros(img.rows, img.cols, CV_8UC1);
}

Mat zeroes_fl(Mat img) {
    return img.zeros(img.rows, img.cols, CV_32FC1);
}

void wait() {
    while (waitKey(30) < 0);
}

void show_and_wait(Mat img) {
    // show image and wait for keyboard input
    imshow(WINDOW_NAME, img);
    wait();
}

Mat compute_gradient_mask(Mat img, bool is_edge) {
    // Returns an image of gradient angles.
    // Edges have smaller convolution kernels because they need to be more detailed
    // Non-edges use integer rather than float matricies to reduce distracting swirling
    // in the image and, as a side effect, speed up the program.

    int sobel_kernel_1;
    Size blur_size_1;

    Mat mask = zeroes_fl(img);
    Mat sobelx, sobely;
    int datatype;

    medianBlur(img, img, 3);

    if (is_edge) {
        sobel_kernel_1 = 7;
        blur_size_1 = Size(3, 3);
        sobelx = zeroes_fl(img);
        sobely = zeroes_fl(img);
        datatype = CV_32FC1;
    }
    else {
        sobel_kernel_1 = SOBEL_KERNEL*2+1;
        blur_size_1 = Size(BLUR_SIZE*2+1, BLUR_SIZE*2+1);
        sobelx = zeroes_int(img);
        sobely = zeroes_int(img);
        datatype = CV_8UC1;
    }

    Sobel(img, sobelx, datatype, 1, 0, sobel_kernel_1);
    Sobel(img, sobely, datatype, 0, 1, sobel_kernel_1);

    boxFilter(sobelx, sobelx, datatype, blur_size_1);
    boxFilter(sobely, sobely, datatype, blur_size_1);

    for (int j=0; j<img.rows; j++) {
        for (int i=0; i<img.cols; i++) {
            if (is_edge)
                mask.at<float>(j,i) = atan2(sobely.at<float>(j,i), sobelx.at<float>(j,i));
            else
                mask.at<float>(j,i) = atan2(sobely.at<uchar>(j,i), sobelx.at<uchar>(j,i));
        }
    }
    return mask;
    }

Point gradient_vector(float grad_val) {
    int dx = (int)(round(cos(grad_val)*LINE_LEN));
    int dy = (int)(round(sin(grad_val)*LINE_LEN));
    return Point(dx, dy);
}

Point perpendicular_vector(float grad_val) {
    int dx = (int)(round(sin(grad_val)*-1*LINE_LEN));
    int dy = (int)(round(cos(grad_val)*LINE_LEN));
    return Point(dx, dy);

}

float rand_float() {
    // returns a float between 0 and 1
    return (float)rand()/RAND_MAX;
}

Point next_point(Mat grad, Point prev, bool is_edge, bool is_perpendicular) {
    // depending on whether we want a point perpendicular or parallel to this
    // one, find it, check the range, and return it
    Point nxt;
    float grad_val = get_fl(grad, prev);
    if (is_edge || is_perpendicular)
        nxt = perpendicular_vector(grad_val);
    else
        nxt = gradient_vector(grad_val);
    nxt += prev;
    if (nxt.x<0 || nxt.x>=grad.cols || nxt.y<0 || nxt.y>=grad.rows)
        throw exception();
    return nxt;
}

vector< vector<Point> > get_gradient_strokes(Mat img, bool is_edge) {
    // Apply our gradient algorithm to draw multiple lines on the image
    // This is the meat of this program
    // Given an image:
    // A: Calculate the gradient
    // B. For every "line_density" pixels
    //  a. Randomly decide whether this will be a perpendicular or gradient line
    //  b. If the image is white at that point or less than our random function, break
    //  c. Extend two connected lines from that point (either along or perpendicular
    //                                                 to the gradient)
    //  d. Create a bezier line from these two lines
    //  e. If we have reached a spot that is too white (by a random function), break
    //  f. Otherwise, continue this line

    vector< vector<Point> > strokes;
    Point p0, p1, p2, dxdy, o_dxdy;
    int line_density = is_edge ? DENSITY/2 : DENSITY+1;

    Mat grad = compute_gradient_mask(img, is_edge);

    for (int y=0; y<img.rows; y+=line_density) {
        for (int x=0; x<img.cols; x+=line_density) {
            bool is_perpendicular = rand()%2 == 0;

            int original_val = get_int(img, Point(x, y));
            if (original_val > WHITE_THRESHOLD+(is_perpendicular-1)*PERP_LESS_THRESHOLD ||
                rand_float() < original_val/255.0)
                    continue;

            int tolerance;
            if (is_edge)
                tolerance = CANNY_TOLERANCE;
            else if (is_perpendicular)
                tolerance = PERPENDICULAR_TOLERANCE;
            else
                tolerance = GRADIENT_TOLERANCE;

            // float grad_val = get_fl(grad,Point(x,y);
            // if (is_edge || is_perpendicular)
            //     o_dxdy = perpendicular_vector(grad_val);
            // else
            //     o_dxdy = gradient_vector(grad_val);

            p0 = Point(x, y);
            int limit = is_edge ? EDGE_LIMIT : GRAD_LIMIT;
            for (int i=0; i<limit; i++) {

                try {
                    p1 = next_point(grad, p0, is_edge, is_perpendicular);
                    p2 = next_point(grad, p1, is_edge, is_perpendicular);
                }
                catch (exception& e) {
                    break;
                }

                // if (i>0 && rads_btwn_vectors(o_dxdy, dxdy) > PI/8.0)
                //     break;

                vector<Point> stroke;
                stroke.push_back(p0);
                for (float t=.2; t<=1.0; t+=.2) {
                    float bez_x = p0.x*(1.-t)*(1.-t) + 2*p1.x*(1.-t)*t + p2.x*t*t;
                    float bez_y = p0.y*(1.-t)*(1.-t) + 2*p1.y*(1.-t)*t + p2.y*t*t;
                    stroke.push_back(Point((int)round(bez_x), (int)round(bez_y)));
                }
                strokes.push_back(stroke);

                int val = get_int(img, p2);
                if (val == WHITE || original_val + tolerance < val)
                    break;
                p0 = p2;
            }
        }
    }
    return strokes;
}

Mat get_canny(Mat img) {
    // The canny edges are crisper and look more like a kid's pencil drawing
    // which is the goal.

    Mat canny = img.clone();
    int height = canny.rows;
    int width = canny.cols;

    // resize to get a thicker line
    resize(canny, canny, Size(width/CANNY_DIV, height/CANNY_DIV));
    Canny(canny, canny, CANNY_LOWER, CANNY_UPPER);
    GaussianBlur(canny, canny, Size(CANNY_GAUSS_BLUR*2+1,CANNY_GAUSS_BLUR*2+1), 0);
    resize(canny, canny, Size(width, height));

    return WHITE - canny;
}

Mat get_laplacian(Mat img) {
    // Laplacian edges look much more realistic but a little less like a drawing

     Mat lap = zeroes_fl(img);
     Laplacian(img, lap, CV_8UC1, LAPLACE*2+1);
     GaussianBlur(lap, lap, Size(5,5), 0);
     return WHITE - lap;
}

Mat draw_strokes_on_img(vector< vector<Point> > strokes, Mat img, Mat ret) {
    // Strokes -> lines on image.
    // Using the color of the original image is kind of cheating but the effect
    // is still there

    Point cur, nxt;
    int color;

    for (int i=0; i<strokes.size(); i++) {
        cur = strokes[i][0];
        for (int j=1; j<strokes[i].size(); j++) {
            nxt = strokes[i][j];
            color = get_int(img, nxt);
            line(ret, cur, nxt, color, 1);
            cur = nxt;
        }
    }
    return ret;

}

Mat apply_fast_gradient(Mat img) {
    // We apply the algorithm in separate ways for the edges and for the rest of
    // the image. This gives us thicker edges that look more like a crosshatch
    // drawing

    Mat ret = img.zeros(img.rows, img.cols, CV_8UC1);
    ret.setTo(WHITE);

    vector< vector<Point> > strokes = get_gradient_strokes(img, false);
    draw_strokes_on_img(strokes, img, ret);

    Mat lap = get_canny(img);
    vector< vector<Point> > lap_strokes = get_gradient_strokes(lap, true);
    draw_strokes_on_img(lap_strokes, img, ret);

    return ret;
}

bool endswith(string const &fullString, string const &ending) {
    // string endswith
    if (fullString.length() >= ending.length())
        return (0 == fullString.compare(fullString.length() - ending.length(),
                                        ending.length(), ending));
    else
        return false;
}

void make_debug_window() {
    // if we want to change some constants
    namedWindow(WINDOW_NAME,1);
    createTrackbar( "DENSITY", WINDOW_NAME, &DENSITY, 10);
    createTrackbar( "BLUR_SIZE", WINDOW_NAME, &BLUR_SIZE, 30);
    createTrackbar( "SOBEL_KERNEL", WINDOW_NAME, &SOBEL_KERNEL, 15);
    createTrackbar( "BLOCK_SIZE_THRESHOLD", WINDOW_NAME, &BLOCK_SIZE_THRESHOLD, 50);
    createTrackbar( "LINE_LEN", WINDOW_NAME, &LINE_LEN, 15);
    createTrackbar( "CANNY_UPPER", WINDOW_NAME, &CANNY_UPPER, 256);
    createTrackbar( "CANNY_LOWER", WINDOW_NAME, &CANNY_LOWER, 256);
    createTrackbar( "LAPLACE", WINDOW_NAME, &LAPLACE, 30);
    createTrackbar( "CANNY_TOLERANCE", WINDOW_NAME, &CANNY_TOLERANCE, 256);
    createTrackbar( "CANNY_GAUSS_BLUR", WINDOW_NAME, &CANNY_GAUSS_BLUR, 10);
    createTrackbar( "CANNY_DIV", WINDOW_NAME, &CANNY_DIV, 10);
}

bool handle_video(VideoCapture cap, bool is_live) {
    if(!cap.isOpened())
        return -1;

    int num_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter outputVideo(OUT_VIDEO_FILENAME, CV_FOURCC('8', 'B', 'P', 'S'),
                            10, Size(frame_width,frame_height), true);

    if (!outputVideo.isOpened()) {
        cout  << "Could not open the output video for write" << endl;
        return -1;
    }

    #ifdef DEBUG
        make_debug_window();
    #endif

    Mat frame, grad, color_frame, bw;
    queue<int> time_queue;
    time_queue.push(time(0));
    for (int i=0; true; i++) {
        cap >> frame;
        if (frame.empty())
            break;

        cvtColor(frame, bw, CV_BGR2GRAY);
        grad = apply_fast_gradient(bw);

        if (is_live) {
            imshow(WINDOW_NAME, grad);
            waitKey(30);
        }
        else {
            cvtColor(grad, color_frame, CV_GRAY2BGR); // mp4 requires a color frame
            outputVideo.write(color_frame);
        }

        time_queue.push(time(0));
        if (time_queue.size() > 10) time_queue.pop();
        float avg_time = ((float)(time_queue.back() - time_queue.front()))/time_queue.size();
        int minutes_remaining = (int)round((num_frames-i)*avg_time/60.0);

        if (!is_live) {
            cout << "frame " << i << "/" << num_frames << endl;
            cout << minutes_remaining << " minutes remaining" << endl << endl;
        }
    }
    if (!is_live)
        cout << "Wrote as " << OUT_VIDEO_FILENAME << endl;
    return 0;
}

void loop_debug(Mat bw) {
    resize(bw, bw, Size(700*bw.rows/bw.cols,700));
    make_debug_window();
    for(;;) {
        Mat grad = apply_fast_gradient(bw);
        imshow(WINDOW_NAME, grad);
        waitKey(1000);
    }
}

int handle_image(Mat bw) {
    cvtColor(bw, bw, CV_BGR2GRAY);
    #ifdef DEBUG
    loop_debug(bw);
    #endif

    bw = apply_fast_gradient(bw);
    imwrite(OUT_IMAGE_FILENAME, bw);
    cout << "Wrote as " << OUT_IMAGE_FILENAME << endl;
    return 0;
}

int main(int argc, char* argv[]) {
    srand (time(NULL));

    // If passed an argument, that argument will be either a video or an image
    if (argc > 1) {
        string in_filename = argv[1];
        if (endswith(in_filename, ".mp4")) {
            VideoCapture cap(in_filename);
            return handle_video(cap, false);
        }
        else {
            Mat bw = imread(in_filename, 1);
            return handle_image(bw);
        }
    }

    // If we are not passed an argument, we are going to do this from the webcam
    VideoCapture cap(0);
    return handle_video(cap, true);
}

