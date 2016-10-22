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
#include <chrono>
#include <mutex>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;


#define BLACK 0
#define WHITE 255
#define PI 3.1415926
#define OUT_VIDEO_FILENAME "out.mp4"
#define WINDOW_NAME "window"
#define DEBUG_WINDOW_WIDTH 700

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
int CANNY_DIV = 2;
int WHITE_THRESHOLD = 180;
int NORMALIZE_BRIGHTNESS_TO = 140;
int PERP_LESS_THRESHOLD = 50;

int num_strokes = 0;

mutex g_pages_mutex;

struct Flags {
    bool laplacian = false;
    bool noequalize = false;
    bool debug = false;
    bool invert = false;
    bool noedges = false;
    bool nofill = false;
    bool strongcorners = false;
    bool crayolaize = false;
    bool saturate = false;
    bool threshold = false;
    bool bw = false;
    bool strokes = false;
    int resize_width = 2000;
    int line_density = 5;
    int brighten = 0;
    string out_filename = "out.png";
    string edges_filename = "";
    string edges_mask_filename = "";
    string fill_filename = "";
    string color_filename = "";
};

int get_int(Mat img, Point p) {
    return img.at<uchar>(p.y, p.x);
}

float get_fl(Mat img, Point p) {
    return img.at<float>(p.y, p.x);
}

Vec3b get_color(Mat img, Point p) {
    return img.at<Vec3b>(p.y, p.x);
}

void set_color(Mat img, Point p, Vec3b color) {
    img.at<Vec3b>(p.y, p.x) = color;
}

void set_int(Mat img, Point p, int val) {
    img.at<uchar>(p.y, p.x) = val;
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

Mat white_int(Mat img) {
    Mat ret = zeroes_int(img);
    ret.setTo(255);
    return ret;
}

Mat zeroes_fl(Mat img) {
    return img.zeros(img.rows, img.cols, CV_32FC1);
}

Mat white_fl(Mat img) {
    Mat ret = zeroes_fl(img);
    ret.setTo(Scalar(255,255,255));
    return ret;
}

float rand_float() {
    // returns a float between 0 and 1
    return (float)rand()/RAND_MAX;
}

int tim() {
    milliseconds ms = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
    return ms.count();
}

void wait() {
    while (waitKey(30) < 0);
}

void show_and_wait(Mat img) {
    // show image and wait for keyboard input
    imshow(WINDOW_NAME, img);
    wait();
}


vector<Vec3b> crayola(24);

void setup_colors() {
    crayola.push_back(Vec3b(237, 10, 63)); // "#ED0A3F" //Red
    crayola.push_back(Vec3b(255, 134, 31)); // #FF3F34 // Red Orange (255, 63, 52)
    crayola.push_back(Vec3b(255, 134, 31));    //  #FF861F  // Orange
    crayola.push_back(Vec3b(251, 232, 112));    //  #FBE870 // Yellow
    crayola.push_back(Vec3b(197, 225, 122));    //  #C5E17A  // Yellow Green
    crayola.push_back(Vec3b(1, 163, 104));    //  #01A368 // Green
    crayola.push_back(Vec3b(118, 215, 234));    //  #76D7EA  // Sky Blue
    crayola.push_back(Vec3b(0, 102, 255)); //0066FF Blue
    crayola.push_back(Vec3b(131, 89, 163));    //  #8359A3  // Violet (Purple)
    crayola.push_back(Vec3b(175, 89, 62));    //  #AF593E  // Brown
    crayola.push_back(Vec3b(0, 0, 0));    //  #000000 //Black
    crayola.push_back(Vec3b(255, 255, 255));    //  #FFFFFF  // White
    crayola.push_back(Vec3b(3, 187, 133)); // Aqua Green #03BB85
    crayola.push_back(Vec3b(255, 223, 0)); // Golden Yellow #FFDF00
    crayola.push_back(Vec3b(139, 134, 128)); // Gray #8B8680
    crayola.push_back(Vec3b(10, 107, 13)); // Jade Green #0A6B0D
    crayola.push_back(Vec3b(143, 216, 216)); // Light Blue #8FD8D8
    crayola.push_back(Vec3b(246, 83, 166)); // Magenta #F653A6
    crayola.push_back(Vec3b(202, 52, 53)); //  Mahogany #CA3435
    crayola.push_back(Vec3b(255, 203, 164)); // Peach #FFCBA4
    crayola.push_back(Vec3b(205, 145, 158)); // Pink #CD919E
    crayola.push_back(Vec3b(250, 157, 90)); // Tan #FA9D5A
    crayola.push_back(Vec3b(163, 111, 64)); // Light Brown #A36F40
    crayola.push_back(Vec3b(255, 174, 66)); // Yellow Orange #FFAE42

    for (int i=0; i<crayola.size(); i++) {
        Mat3f one_pixel(1, 1, crayola[i]);
        cvtColor(one_pixel, one_pixel, CV_RGB2HSV);
        crayola[i] = get_color(one_pixel, Point(0, 0));
    }
}

Vec3b best_color(Vec3b color) {
    int best_idx = -1;
    int best_val = 255;
    for (int i=0; i<crayola.size(); i++) {
        int val = abs(color[0] - crayola[i][0]);
        if (val > 128)
            val = 256-val;
        if (val < best_val) {
            best_idx = i;
            best_val = val;
        }

    }

    return crayola[best_idx];
}

void crayolaize(Mat color_img, Flags f) {
    cvtColor(color_img, color_img, CV_RGB2HSV);
    vector<Mat> channels(3);
    split(color_img, channels);

    Vec3b red = Vec3b(251, 232, 112);

    for (int j=0; j<color_img.rows; j++) {
        for (int i=0; i<color_img.cols; i++) {
            Vec3b color = get_color(color_img, Point(i, j));
            color[0] = (color[0]/7)*7; // 36 colors
            if (color[0] > 255)
                color[0] = 255;
            color[2] = (color[2]/50)*50; // 5 shades
            if (f.saturate) {
                if (color[1] > 128 && color[2] < 128) {
                    int diff = 255 - color[1];
                    color[1] = 255;
                    color[2] += diff;

                }
            }
            set_color(color_img, Point(i,j), color);
        }
    }
    cvtColor(color_img, color_img, CV_HSV2RGB);
}

Mat apply_threshold(Mat bw) {
    Mat mask = zeroes_int(bw);
    Mat ret = white_int(bw);
    adaptiveThreshold(bw, mask, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, bw.cols/10+1, 0);
    bw.copyTo(ret, mask);
    return ret;
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

            if (mask.at<float>(j,i) == 0)
                mask.at<float>(j,i) = PI/2 + (rand_float()-1)/2;
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

void draw_stroke_on_img(vector<Point> stroke, Mat color, Mat ret, Flags f) {
    Point cur, nxt;
    cur = stroke[0];
    Scalar the_color = Scalar(0,0,0);
    for (int j=1; j<stroke.size(); j++) {
        nxt = stroke[j];
        if (f.bw) {
            line(ret, cur, nxt, the_color, 1);
        }
        else {
            Vec3b the_color = get_color(color, nxt);
            line(ret, cur, nxt, Scalar(the_color.val[0], the_color.val[1], the_color.val[2]), 1);
        }
        cur = nxt;
    }
}

class Parallel_process : public ParallelLoopBody {

    private:
        Mat img;
        Mat grad;
        Mat color;
        Mat &ret;
        vector<vector<Point>> all_strokes;
        bool is_edge;
        int line_density;
        Flags f;
        // vector< vector<Point> > &strokes;

    public:
        Parallel_process(Mat _img, Mat _grad, Mat _color, Mat &_ret, bool _is_edge, int _line_density, vector<vector<Point>> _all_strokes, Flags _f)
            : img(_img), grad(_grad), color(_color), ret(_ret), is_edge(_is_edge), line_density(_line_density), all_strokes(_all_strokes), f(_f){}

        virtual void operator()(const Range& range) const {
            for(int _y = range.start; _y < range.end; _y++) {
                for(int x = 0; x < img.cols; x+=line_density) {
                    int y = _y*line_density;
                    // printf("%d,%d\n", x, y);
                    Point p0, p1, p2, dxdy, o_dxdy;
                    bool is_perpendicular = rand()%2 == 0;

                    int original_val = get_int(img, Point(x, y));
                    if (original_val > WHITE_THRESHOLD+(is_perpendicular-1)*PERP_LESS_THRESHOLD ||
                        rand_float() < original_val/255.0)
                            continue;

                    if (f.strongcorners) {
                        int middle_x = img.cols/2;
                        int middle_y = img.rows/2;
                        float norm = (float)(middle_x*middle_y);
                        if (pow((middle_y-y)*(middle_x-x)/norm,2) > pow(rand_float(), 2))
                            continue;
                    }

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
                    // int limit = is_edge ? 10-line : GRAD_LIMIT;
                    int limit = 10-line_density;
                    for (int i=0; i<5; i++) {

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
                        vector<vector<Point>> all_strokes_2;

                        stroke.push_back(p0);
                        for (float t=.2; t<=1.0; t+=.2) {
                            float bez_x = p0.x*(1.-t)*(1.-t) + 2*p1.x*(1.-t)*t + p2.x*t*t;
                            float bez_y = p0.y*(1.-t)*(1.-t) + 2*p1.y*(1.-t)*t + p2.y*t*t;
                            stroke.push_back(Point((int)bez_x, (int)bez_y));
                        }

                        // if (f.strokes) {
                        //     #pragma omp critical
                        //     all_strokes.insert(all_strokes.end(), all_strokes_2.begin(), all_strokes_2.end())
                        //     // all_strokes.push_back(stroke);
                        // }
                        // else {
                        num_strokes += 1;
                        draw_stroke_on_img(stroke, color, ret, f);
                        // }

                        int val = get_int(img, p2);
                        if (val == WHITE || original_val + tolerance < val)
                            break;
                        p0 = p2;
                    }
                }
            }
        }
    };


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

Mat get_l_free(Mat img) {
    Mat ret = img.zeros(img.rows, img.cols, CV_8UC1);
    Mat lab = img.clone();
    cvtColor(lab, lab, CV_BGR2HSV);
    Mat planes[3];
    split(lab, planes);
    for(int y = 0; y < lab.rows; y++) {
        for(int x = 0; x < lab.cols; x++) {
            int l = get_int(planes[2], Point(x, y));
            int a = get_int(planes[1], Point(x, y));
            int b = get_int(planes[0], Point(x, y));
            set_int(ret, Point(x, y), (a+b+l/5)/3);
        }
    }
    equalizeHist(ret, ret);
    GaussianBlur(ret, ret, Size(CANNY_GAUSS_BLUR*2+1,CANNY_GAUSS_BLUR*2+1), 5);
    medianBlur(ret, ret, 5);
    return ret;
}

Mat get_canny(Mat img) {
    // The canny edges are crisper and look more like a kid's pencil drawing
    // which is the goal.

    Mat canny = img.clone();
    int height = img.rows;
    int width = img.cols;

    resize(canny, canny, Size(width/CANNY_DIV, height/CANNY_DIV));
    canny = get_l_free(canny);
    // imwrite("lfree.png", canny);
    Canny(canny, canny, CANNY_LOWER, CANNY_UPPER);
    GaussianBlur(canny, canny, Size(CANNY_GAUSS_BLUR*2+1,CANNY_GAUSS_BLUR*2+1), 0);
    resize(canny, canny, Size(width, height));
    // imwrite("canny.png", WHITE - canny);
    return WHITE - canny;
}

Mat get_laplacian(Mat img) {
    // Laplacian edges look much more realistic but a little less like a drawing

     Mat lap = zeroes_fl(img);
     Laplacian(img, lap, CV_8UC1, LAPLACE*2+1);
     GaussianBlur(lap, lap, Size(5,5), 0);
     return WHITE - lap;
}

// Mat draw_strokes_on_img(vector< vector<Point> > strokes, Mat color, Mat ret) {
//     // Strokes -> lines on image.
//     // Using the color of the original image is kind of cheating but the effect
//     // is still there

//     for (int i=0; i<strokes.size(); i++) {
//         draw_stroke_on_img(strokes[i], color, ret);
//     }
// }

void decrease_contrast(Mat img, float mult) {
    int add = (int)255.0*(1.0-mult)/2.0;
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            Vec3b c = get_color(img, Point(x, y));
            set_color(img, Point(x, y), Vec3b(c.val[0]*mult+add, c.val[1]*mult+add, c.val[2]*mult+add));
        }
    }
}

void brighten(Mat img, int add) {
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            Vec3b c = get_color(img, Point(x, y));
            int ac = c.val[0] + add;
            int bc = c.val[1] + add;
            int cc = c.val[2] + add;
            if (0 > ac) ac = 0;
            if (ac > 255) ac = 255;
            if (0 > bc) bc = 0;
            if (bc > 255) bc = 255;
            if (0 > cc) cc = 0;
            if (cc > 255) cc = 255;
            set_color(img, Point(x, y), Vec3b(ac, bc, cc));
        }
    }
}


void apply_contrast_equalization(Mat color) {
    cvtColor(color, color, CV_RGB2HSV);
    vector<Mat> planes(3);
    split(color, planes);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(planes[2], planes[2]);

    int mean_brightness = (int)mean(planes[2])[0];
    cout << mean_brightness << endl;
    int add = NORMALIZE_BRIGHTNESS_TO - mean_brightness;

    // for(int y = 0; y < color.rows; y++) {
    //     for(int x = 0; x < color.cols; x++) {
    //         int ac = get_int(planes[2], Point(x, y)) + add;
    //         if (0 > ac) ac = 0;
    //         if (ac > 255) ac = 255;
    //         set_int(planes[2], Point(x, y), ac);
    //     }
    // }
    planes[2].convertTo(planes[2], -1, 1.0, add);

    // TODO this is might bea a hue-shift
    // for(int y = 0; y < dst.rows; y++) {
    //     for(int x = 0; x < dst.cols; x++) {
    //         set_int(planes[2], Point(x, y), abs(get_int(planes[2], Point(x, y))-30));
    //     }
    // }

    merge(planes, color);
    cvtColor(color, color, CV_HSV2RGB);
}

Mat resize_to_width(Mat img, int width) {
    float rows_per_col = ((float)img.rows) / ((float)img.cols);
    resize(img, img, Size(width, (int)(width*rows_per_col)));
    return img;
}


void parallel_it(Mat bw, Mat grad, Mat color, Mat ret, bool is_edge, int line_density, vector<vector<Point>> all_strokes, Flags f) {
    if (f.bw) {
        line_density*=3/4;
    }
    parallel_for_(Range(0, bw.rows/line_density), Parallel_process(bw, grad, color, ret, is_edge, line_density, all_strokes, f));
}


Mat apply_fast_gradient(Mat color, Flags f) {
    // We apply the algorithm in separate ways for the edges and for the rest of
    // the image. This gives us thicker edges that look more like a crosshatch
    // drawing

    if (!f.noequalize)
        apply_contrast_equalization(color);

    if (f.crayolaize) {
        crayolaize(color, f);
        // imwrite("crayola.png", color);
    }

    if (f.brighten) {
        brighten(color, f.brighten);
    }

    color = resize_to_width(color, f.resize_width);

    if (f.color_filename.length()) {
        imwrite(f.color_filename, color);
    }

    Mat bw = color.zeros(color.rows, color.cols, CV_8UC1);
    cvtColor(color, bw, CV_BGR2GRAY);

    if (f.threshold)
        bw = apply_threshold(bw);

    int a, b;
    Mat grad, grad2;

    Mat ret = bw.zeros(color.rows, color.cols, CV_8UC3);
    ret.setTo(WHITE);

    if (f.invert)
        bitwise_not(bw, bw);

    vector<vector<Point>> all_strokes;

    if (!f.noedges) {
        a = tim();
        Mat edges = f.laplacian ? get_laplacian(bw) : get_canny(color);
        if (f.edges_mask_filename.length())
            imwrite(f.edges_mask_filename, edges);
        grad = compute_gradient_mask(edges, true);
        if (f.edges_filename.length()){
            grad.convertTo(grad2, CV_8UC1, (int)(255/6.28));
            imwrite(f.edges_filename, grad2);
        }
        else{
            int line_density = f.line_density/2;
            parallel_it(edges, grad, color, ret, true, line_density, all_strokes, f);
        }
        b = tim();
    }

    if (!f.nofill) {
        a = tim();
        grad = compute_gradient_mask(bw, false);
        if (f.fill_filename.length()){
            grad.convertTo(grad2, CV_8UC1, (int)(255/6.28));
            imwrite(f.fill_filename, grad2);
        }
        else{
            int line_density = f.line_density+1;
            parallel_it(bw, grad, color, ret, false, line_density, all_strokes, f);
        }
        b = tim();
    }

    cout << num_strokes << endl;

    if (f.strokes) {
        cout << all_strokes.size() << endl;
        exit(0);
    }

    return ret;
}

// void white_edges_corner() {
//     GaussianBlur(edges, edges, Size(CANNY_GAUSS_BLUR*5+1,CANNY_GAUSS_BLUR*5+1), 0);
//     threshold(edges, edges, 240, 255,THRESH_BINARY);
//     Mat colors2 = img.zeros(color.rows, color.cols, CV_8UC3);
//     colors2.setTo(WHITE);
//     imwrite("edges.png", edges);
//     // bitwise_and(color,color,color,edges);
//     color.copyTo(colors2, edges);
//     imwrite("mask.png", colors2);

// }

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
    createTrackbar( "BLUR_SIZE", WINDOW_NAME, &BLUR_SIZE, 30);
    createTrackbar( "SOBEL_KERNEL", WINDOW_NAME, &SOBEL_KERNEL, 15);
    createTrackbar( "BLOCK_SIZE_THRESHOLD", WINDOW_NAME, &BLOCK_SIZE_THRESHOLD, 50);
    createTrackbar( "LINE_LEN", WINDOW_NAME, &LINE_LEN, 15);
    createTrackbar( "CANNY_UPPER", WINDOW_NAME, &CANNY_UPPER, 256);
    createTrackbar( "CANNY_LOWER", WINDOW_NAME, &CANNY_LOWER, 256);
    createTrackbar( "LAPLACE", WINDOW_NAME, &LAPLACE, 30);
    createTrackbar( "CANNY_TOLERANCE", WINDOW_NAME, &CANNY_TOLERANCE, 256);
    createTrackbar( "CANNY_GAUSS_BLUR", WINDOW_NAME, &CANNY_GAUSS_BLUR, 20);
    createTrackbar( "CANNY_DIV", WINDOW_NAME, &CANNY_DIV, 10);
}


bool handle_video(VideoCapture cap, Flags f, bool is_live) {

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

    if (f.debug) {
        make_debug_window();
        is_live = true;
    }

    Mat color, grad, color_out, bw;

    cap >> color;
    imshow(WINDOW_NAME, color);
    cvSetWindowProperty(WINDOW_NAME, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    queue<int> time_queue;
    if (!is_live) time_queue.push(tim());


    while (true) {
        // for (int i=0; i<num_frames; i++) {

        // TODO. Took this out to make it run live. Probably need to put it back in
        // to handle video
        // if (i && !(i%50)) {
        //     // QTKit starts to run slowly the longer the film on a Mac
        //     // This hack takes care of the problem
        //     VideoCapture cap(in_filename);
        //     for (int j=0; j<i; j++) {
        //         cap >> frame;
        //         cout << "frame" << endl;
        //     }

        // }
        int a = tim();
        cap >> color;
        if (color.empty()) {
            continue;
        }
        if (is_live) {
            // Flip along y axis for mirror effect
            flip(color, color, 1);
        }

        int b = tim();


        grad = apply_fast_gradient(color, f);

        if (is_live) {
            a = tim();
            imshow(WINDOW_NAME, grad);
            waitKey(1);
            b = tim();
        }
        else {
            cvtColor(grad, color_out, CV_GRAY2BGR); // mp4 requires a color frame
            outputVideo.write(color_out);

            // int minutes_remaining = (int)round((num_frames-i)*avg_time/60.0);
            // cout << "frame " << i << "/" << num_frames << endl;
            // cout << minutes_remaining << " minutes remaining" << endl;
        }

        time_queue.push(tim());
        if (time_queue.size() > 10) time_queue.pop();
        long avg_time = ((time_queue.back() - time_queue.front()))/(time_queue.size());

        cout << avg_time << endl;
    }
    if (!is_live)
        cout << "Wrote as " << OUT_VIDEO_FILENAME << endl;
    return 0;
}


void loop_debug(Mat color, Flags f) {
    make_debug_window();
    for(;;) {
        Mat grad = apply_fast_gradient(color, f);
        imshow(WINDOW_NAME, grad);
        waitKey(1000);
    }
}


int handle_image(Mat color, Flags f) {
    if (f.debug)
        loop_debug(color, f);

    color = apply_fast_gradient(color, f);
    if (!f.color_filename.length())
        imwrite(f.out_filename, color);
    cout << "Wrote as " << f.out_filename << endl;
    return 0;
}

int main(int argc, char* argv[]) {
    srand (time(NULL));

    string in_filename = "";
    Flags flags;

    // Loop over arguments
    for (int i=1; i<argc; i++) {
        string arg = string(argv[i]);
        if (arg.compare("--laplacian") == 0) {
            flags.laplacian = true;
            continue;
        }
        if (arg.compare("--noequalize") == 0) {
            flags.noequalize = true;
            continue;
        }
        if (arg.compare("--debug") == 0) {
            flags.debug = true;
            continue;
        }
        if (arg.compare("--invert") == 0) {
            flags.invert = true;
            continue;
        }
        if (arg.compare("--noedges") == 0) {
            flags.noedges = true;
            continue;
        }
        if (arg.compare("--nofill") == 0) {
            flags.nofill = true;
            continue;
        }
        if (arg.compare("--strongcorners") == 0) {
            flags.strongcorners = true;
            continue;
        }
        if (arg.compare("--crayolaize") == 0) {
            flags.crayolaize = true;
            continue;
        }
        if (arg.compare("--saturate") == 0) {
            flags.saturate = true;
            continue;
        }
        if (arg.compare("--threshold") == 0) {
            flags.threshold = true;
            continue;
        }
        if (arg.compare("--bw") == 0) {
            flags.bw = true;
            continue;
        }
        if (arg.compare("--strokes") == 0) {
            flags.strokes = true;
            continue;
        }
        if (arg.compare("--out") == 0) {
            flags.out_filename = string(argv[i+1]);
            i++;
            continue;
        }
        if (arg.compare("--edges-filename") == 0) {
            flags.edges_filename = string(argv[i+1]);
            i++;
            continue;
        }
        if (arg.compare("--edges-mask-filename") == 0) {
            flags.edges_mask_filename = string(argv[i+1]);
            i++;
            continue;
        }
        if (arg.compare("--fill-filename") == 0) {
            flags.fill_filename = string(argv[i+1]);
            i++;
            continue;
        }
        if (arg.compare("--color-filename") == 0) {
            flags.color_filename = string(argv[i+1]);
            i++;
            continue;
        }
        if (arg.compare("--resize") == 0) {
            flags.resize_width = stoi(string(argv[i+1]));
            i++;
            continue;
        }
        if (arg.compare("--line-density") == 0) {
            flags.line_density = stoi(string(argv[i+1]));
            i++;
            continue;
        }
        if (arg.compare("--brighten") == 0) {
            flags.brighten = stoi(string(argv[i+1]));
            i++;
            continue;
        }


        in_filename = arg;
    }
    setup_colors();
    // We can either process a video or an image
    if (in_filename.length()) {
        if (endswith(in_filename, ".mpg") || endswith(in_filename, ".mp4") ||
            endswith(in_filename, ".gif") || endswith(in_filename, ".mov")) {
                VideoCapture cap(in_filename);
                return handle_video(cap, flags, false);
        }
        else {
            Mat color = imread(in_filename, 1);
            return handle_image(color, flags);
        }
    }

    // If we are not passed a file, use the webcam
    VideoCapture cap(0);
    return handle_video(cap, flags, true);
}

