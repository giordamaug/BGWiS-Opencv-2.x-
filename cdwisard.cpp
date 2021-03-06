#include <cdwisard.hpp>

char buffer[1024];
// getTextSizestring 
const string WinTitle = "CD Wisard (V.1.0)";
// url
string CamUrl, VideoUrl;

// Colors
Scalar bgcolor = Scalar(106,117,181);
// Gui settings

void rgb2rgb(Mat in, Mat &out) {
    in.copyTo(out);
}

void rgb2lab(Mat in, Mat &out) {
    cvtColor(in, out, CV_BGR2Lab);
}

void rgb2hsv(Mat in, Mat &out) {
    cvtColor(in, out, CV_BGR2HSV);
}

void lab2rgb(Mat in, Mat &out) {
    cvtColor(in, out, CV_Lab2BGR);
}

void hsv2rgb(Mat in, Mat &out) {
    cvtColor(in, out, CV_HSV2BGR);
}

void rgb2gray(Mat in, Mat &out) {
    cvtColor(in, out, CV_BGR2GRAY);
}

void lab2gray(Mat in, Mat &out) {
    cvtColor(in, out, CV_Lab2BGR);
    cvtColor(out, out, CV_BGR2GRAY);
}

void hsv2gray(Mat in, Mat &out) {
    cvtColor(in, out, CV_HSV2BGR);
    cvtColor(out, out, CV_BGR2GRAY);
}

int getdir(string dir, vector<string> &files, string ext) {
    DIR *dp;
    struct dirent *dirp;
    int cnt=0;
    string fn;
    
    if((dp = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << -1 << ") wrong dir " << dir << endl;
        return -1;
    }
    while ((dirp = readdir(dp)) != NULL) {
        if (dirp->d_name[0] != '.') {  /* ignore hidden files */
            fn = string(dirp->d_name);
            if(fn.substr(fn.find_last_of(".") + 1) == ext) {
                files.push_back(fn);
                cnt++;
            }
        }
    }
    closedir(dp);
    if (cnt == 0) {
        cout << "Error(" << -2 << ") empty dir " << dir << endl;
        return -2;
    }
    return cnt;
}

Mat& discretize(Mat& I, int dimTics) {
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));
    
    int channels = I.channels();
    
    int nRows = I.rows;
    int nCols = I.cols * channels;
    
    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    
    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = (unsigned char) ((p[j] / dimTics) * dimTics);
;
        }
    }
    return I;
}

void showImagesOld(Mat dispimg, int w, int h, Mat img1, Rect roi1, Mat img2, Rect roi2, Mat img3, Rect roi3,
		double fps, double rfps, Point pos) {
    int baseline;
    Size tsize = getTextSize("dummy",CV_FONT_HERSHEY_PLAIN,1.0,1,&baseline);
    img1.copyTo(dispimg(roi1));
    img2.copyTo(dispimg(roi2));
    img3.copyTo(dispimg(roi3));
    sprintf(buffer,"%-15s%.0f", "SOURCE FPS:", fps);
    putText(dispimg,buffer,Point(pos.x,pos.y+tsize.height+5),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255,255,255));
    sprintf(buffer,"%-15s%.0f", "PROCESS FPS:", rfps);
    putText(dispimg,buffer,Point(pos.x,pos.y+2*(tsize.height+5)),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255,255,255));
    imshow(WinTitle,dispimg);
}

void showImages(Mat &dispimg, list< pair < Mat, Rect > > imglist, list< pair < string, Point > > tlist) {
    int baseline;
    Size tsize = getTextSize("dummy",CV_FONT_HERSHEY_PLAIN,1.0,1,&baseline);
    for (std::list< pair < Mat, Rect > >::const_iterator it = imglist.begin(); it != imglist.end(); ++it) {
        (*it).first.copyTo(dispimg((*it).second));
    }
    for (std::list< pair < string, Point > >::const_iterator it = tlist.begin(); it != tlist.end(); ++it) {
        putText(dispimg,(*it).first,(*it).second,CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255,255,255));
    }
    //putText(dispimg,buffer,Point(pos.x,pos.y+2*(tsize.height+5)),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255,255,255));
    imshow(WinTitle,dispimg);
}

int main(int argc, char** argv) {
    // Parse commands globs
    DIR *dp, *op;   // input dir
    FILE *fp;
    int dcnt;
    vector<string> dlist = vector<string>();
    string indirname;
    string outdirname;
    string bgoutdirname;
    string videoname;
    string gtfilename;
    int incr, decr;
    int err;
    int delta;
    bool outflag= false, bgoutflag=false, gtflag=false;
    
    // Graphics globs
    Mat frame, frame_orig; //current frame
    int frameidx = 0;
    Mat bgmodel, bgmodel_out; //fg/bg masks  generated by MOG2 method
    Mat frameYCrCb, bgmodelYCrCb; //fg/bg masks  generated by MOG2 method
    Mat tmpMask, dframe, deltaimg;
    Mat frameY[3], bgmodelY[3];
    list< pair < Mat, Rect > > imglist; // image/ROI list for display
    list< pair < string, Point > > titlelist; // image/ROI list for display
    void (*convert)(Mat, Mat &);
    void (*backconvert)(Mat, Mat &);
    string titleupleft, titleupright="BG Model", titledownleft="Diff |Input-BG|", titledownright="FG Detection";

    // Timing globs
    double fps, rfps, mfps=-2;
    struct timeval t1, t2;
    
    // Set Command Line Parser
    CmdLine cmd("CD Wisard - command description message", ' ', "1.0");
    SwitchArg verboseSwitch("v", "verbose", "show configuration", false);
    ValueArg<string> inputdirArg("d", "inputdir", "Input video folder", true, "", "directory");
    ValueArg<string> outputdirArg("o", "outputdir", "Output folder", false, "", "directory");
    ValueArg<string> gtfileArg("g", "gtfile", "GT file name", false, "", "filename");
    ValueArg<string> bgdirArg("O", "bgdir", "BG Output folder", false, "", "directory");
    ValueArg<string> colorcodingArg("m", "colorcode", "Color conding (Lab)", false, "Lab", "string");
    ValueArg<int> nbitArg("b","nbit","bit resolution (4)",false,4,"bitno");
    ValueArg<int> ticsArg("n","tics","tics in color scale (16)",false,16,"ticsno");
    ValueArg<int> cachesizeArg("c", "cachesize", "cache size (20)", false, 20,"size");
    ValueArg<string> policyArg("p", "policy", "train policy (+1, -1)", false, "1:1","int:int");
    ValueArg<double> watermarkArg("w", "watermark", "bleaching threshold (0)", false, 0.0,"threshold");
    ValueArg<double> uppermarkArg("u", "uppermark", "upper mark for ram contents (50)", false, 50.0,"size");
    ValueArg<double> selectArg("y", "select", "selection threshold (3)", false, 3,"size");
    ValueArg<int> learnArg("l", "learn", "learning boots time (0)", false, 0,"size");
    SwitchArg reverseSwitch("R", "reverse", "reverse mode (diasbled)", false);
    ValueArg<double> threshArg("t", "threshold", "classification threshold (0.75)", false, 0.75,"threshold");
    SwitchArg blurFlag("B", "blur", "enable blur (disabled)", false);
    ValueArg<string> extArg("x", "extension", "image extension [png]", false, "png", "extension");
    cmd.add(verboseSwitch);
    cmd.add(reverseSwitch);
    cmd.add(inputdirArg);
    cmd.add(outputdirArg);
    cmd.add(gtfileArg);
    cmd.add(bgdirArg);
    cmd.add(colorcodingArg);
    cmd.add(nbitArg);
    cmd.add(ticsArg);
    cmd.add(cachesizeArg);
    cmd.add(policyArg);
    cmd.add(watermarkArg);
    cmd.add(uppermarkArg);
    cmd.add(learnArg);
    cmd.add(selectArg);
    cmd.add(threshArg);
    cmd.add(blurFlag);
    cmd.add(extArg);
    // Parse arguments
    cmd.parse(argc, argv);
    
    indirname = inputdirArg.getValue();
    int pos;
    if ((pos = indirname.find_last_of("/\\")) == indirname.size() - 1) {
        indirname = indirname.substr(0,indirname.size()-1);
        pos = indirname.find_last_of("/\\");
    }
    videoname = indirname.substr(pos+1);
    if ((dp = opendir (indirname.c_str())) == NULL) {
        cout << "Could not open input dir" << endl;
        exit(-1);
    }
    if ((dcnt = getdir(indirname.c_str(), dlist, extArg.getValue())) < 0) {
        cout << "Empty input dir" << endl;
        exit(-1);
    }
    // shuffle frame
    //std::random_shuffle ( dlist.begin(), dlist.end() );
    if ((err = parsePolicy(policyArg.getValue(),incr,decr)) < 0) {
        cerr << "Parse error: Argument: -w" << endl;
        cerr << string(13, ' ') << "window setting must be <int>:<int>" << endl;
        exit(-1);
    }
    if (outputdirArg.isSet()) {
        outdirname = outputdirArg.getValue();
        if ((op = opendir (outdirname.c_str())) == NULL) {
            cout << "Warn: Could not open output dir" << endl;
        } else outflag = true;
    }
    if (gtfileArg.isSet()) {
        gtfilename = gtfileArg.getValue();
        dframe = imread(gtfilename, CV_LOAD_IMAGE_COLOR);
        if(! dframe.data ) {
            cout << "Warn: Could not open GT file" << endl;
        } else {
            titledownleft="Diff |GT-BG|";
            gtflag = true;
        }
    }
    if (bgdirArg.isSet()) {
        bgoutdirname = bgdirArg.getValue();
        if ((op = opendir (bgoutdirname.c_str())) == NULL) {
            cout << "Warn: Could not open BG output dir" << endl;
        } else bgoutflag = true;
    }
    string coding;
    if (colorcodingArg.isSet()) {
        coding = colorcodingArg.getValue();
        if (colorcodingArg.getValue() == "RGB") {
            convert = &rgb2rgb;
            backconvert = &rgb2rgb;
            titleupleft = "Input(RGB): ";
        } else if (colorcodingArg.getValue() == "Lab") {
            convert = &rgb2lab;
            backconvert = &lab2rgb;
            titleupleft = "Input(Lab): ";
        } else if (colorcodingArg.getValue() == "HSV") {
            convert = &rgb2hsv;
            backconvert = &hsv2rgb;
            titleupleft = "Input(HSV): ";
        } else {
            cerr << "Parse error: Argument: -m" << endl;
            cerr << string(13, ' ') << "Color space must be RGB|HSV|Lab" << endl;
            exit(-1);
        }
    } else {
        coding = colorcodingArg.getValue();
        convert = &rgb2lab;
        backconvert = &lab2rgb;
        titleupleft = "Input(Lab): ";
    }
    
    delta = 256 / ticsArg.getValue();
    // Get firt frame
    frame_orig = imread(indirname + "/" + dlist[frameidx]);
    if(! frame_orig.data ) {
        cout << "Could not open read frame" << endl;
        exit(-1);
    }
    convert(frame_orig, frame);
    int w = frame.cols;
    int h = frame.rows;
    cout << "Processing Video (" << w << "x" << h << ")" << endl;

    ///*
    BackgroundSubtractorWIS Subtractor;
    Subtractor.set("noBits", nbitArg.getValue());
    Subtractor.set("noTics", ticsArg.getValue());
    Subtractor.set("trainIncr", incr);
    Subtractor.set("trainDecr", decr);
    Subtractor.set("cacheSize", cachesizeArg.getValue());
    Subtractor.set("varWatermark", watermarkArg.getValue());
    Subtractor.set("varThreshold", threshArg.getValue());
    Subtractor.set("varUpWatermark", uppermarkArg.getValue());
    Subtractor.set("selectThreshold", selectArg.getValue());
    Subtractor.set("learningStage", learnArg.getValue());
    Subtractor.initialize(frame.size(), frame.type());
    //*/
    if (verboseSwitch.isSet()) {
        cout << left << setw(MAXWIDTH) << setfill(filler) << "I/O GRAPHICS PARAMS" << endl;
        cout << left << setw(MAXWIDTH) << setfill(filler) << "Video Coding" << ": " << coding << endl;
        cout << left << setw(MAXWIDTH) << setfill(filler) << "Input Directory" << ": " << indirname << endl;
        cout << left << setw(MAXWIDTH) << setfill(filler) << "Video Name" << ": " << videoname << endl;
        cout << left << setw(MAXWIDTH) << setfill(filler) << "Blur" << ": " << blurFlag.isSet() << endl;
        cout << left << setw(MAXWIDTH) << setfill(filler) << "Reverse" << ": " << (reverseSwitch.isSet() ? "ON" : "OFF")  << endl;
        cout << left << setw(MAXWIDTH) << setfill(filler) << "WISARD DETECTOR PARAMS" << endl;
        Subtractor.printinfo(MAXWIDTH);
    }
    
    // Open Video Stream and get properties
    fps = 30;
    mfps = (int)fps;
    // Create output display and its geometry
    //int hskip = 20, wskip = 40;
    int hskip = 16, wskip = 4;
    int dcols = 2;
    Mat outFrame(h+2*hskip+wskip,w*dcols+(dcols+1)*wskip,CV_8UC3,bgcolor);
    
    // create window an put icons
    cvNamedWindow(WinTitle.c_str(),CV_WINDOW_AUTOSIZE);
    
    int cnt = 0;
    // video processing loop
    int plus = 1;
    
    while (frameidx >= 0 and frameidx < dcnt) {
        gettimeofday(&t1,NULL);
        frame_orig = imread(indirname + "/" + dlist[frameidx]);  //get one frame form video
        if(! frame_orig.data ) {
            cout << "Could not open read frame" << endl;
            exit(-1);
        }
        if (blurFlag.isSet()) blur(frame,frame,Size(3,3));
        convert(frame_orig, frame);

        Subtractor.operator()(frame, tmpMask);  // apply change detection
        Subtractor.getBackgroundImage(bgmodel); // get Background Model
        
        if (waitKey(30) >= 0)
            break;
        gettimeofday(&t2,NULL);
        rfps = (int)(1.0 / ((double) (t2.tv_usec - t1.tv_usec)/1000000 + (double) (t2.tv_sec - t1.tv_sec)));
        if (mfps != rfps && frameidx % 5 == 0) mfps = rfps;
        
        if (!gtflag) {
            dframe = discretize(frame,Subtractor.dimTics);
        } else dframe = discretize(dframe,Subtractor.dimTics);
        // difference by luminance (Ycrcb)
        backconvert(frame,frame);
        backconvert(bgmodel, bgmodel);

        outFrame.setTo(bgcolor);
        
        imglist.clear();
        titlelist.clear();
        imglist.push_back(make_pair(frame_orig,Rect(Point(wskip,hskip-2),Size(w,h))));  // original frame (Up left)
        titlelist.push_back(make_pair(titleupleft + format("%05d",frameidx),Point(wskip,hskip-5)));
        imglist.push_back(make_pair(bgmodel,Rect(Point(wskip*2+w,hskip-2),Size(w,h))));  // bgmodel (Up right)
        titlelist.push_back(make_pair(titleupright,Point(wskip*2+w,hskip-5)));
        showImages(outFrame, imglist, titlelist);
        //if (outflag) imwrite(outdirname + format("/BC_%06d.png",frameidx), bgmodel);
        if (outflag) imwrite(outdirname + format("/out_%06d.png",frameidx), outFrame);
        frameidx += plus;
        if (reverseSwitch.isSet() and frameidx == dcnt) { plus = -1; frameidx--; frameidx--; };  // reverse
    }
    if (bgoutflag) imwrite(bgoutdirname + format("/BC_%s.png",videoname.c_str()), bgmodel);
}


