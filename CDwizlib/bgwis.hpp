//
//  BackgrounSubtractorWIS.hpp
//  
//
//  Created by Maurizio Giordano on 24/04/15.
//
//


#include "opencv2/core/core.hpp"
#include "opencv2/video/background_segm.hpp"
// Wisard include
#include "wisard.hpp"

using namespace std;
using namespace cv;
long int wcounter;

#define CV_INIT_ALGORITHM(classname, algname, memberinit) \
static ::cv::Algorithm* create##classname() \
{ \
return new classname; \
} \
\
static ::cv::AlgorithmInfo& classname##_info() \
{ \
static ::cv::AlgorithmInfo classname##_info_var(algname, create##classname); \
return classname##_info_var; \
} \
\
static ::cv::AlgorithmInfo& classname##_info_auto = classname##_info(); \
\
::cv::AlgorithmInfo* classname::info() const \
{ \
static volatile bool initialized = false; \
\
if( !initialized ) \
{ \
initialized = true; \
classname obj; \
memberinit; \
} \
return &classname##_info(); \
}

namespace cv {
    static const int defaultNoBits = 4;
    static const int defaultNoTics = 16;
    static const int defaultCacheSize = 32;
    static const double defaultTrainIncr = 1.0;
    static const double defaultTrainDecr = 1.0;
    static const double defaultVarThreshold = 0.75;
    static const double defaultVarWatermark = 0.0;
    static const double defaultVarUpWatermark = 50.0;
    static const double defaultSelectThreshold = 3;
    static const int defaultLearningStage = 0;

    class CV_EXPORTS_W BackgroundSubtractorWIS : public BackgroundSubtractor
    {
    public:
        int noBits;
        int noTics;
        int noRams;
        int dimTics;
        double trainIncr;
        double trainDecr;
        double varThreshold;
        double varWatermark;
        int selectThreshold;
        int learningStage;
        wvalue_t varUpWatermark;
        int frameNum_;
        int cacheSize;
        unsigned long int hits, misses, tcount;
        Size frameSize;
        int frameType;
        
        CV_WRAP BackgroundSubtractorWIS();
        CV_WRAP BackgroundSubtractorWIS(int noBits, int noTics);
        CV_WRAP BackgroundSubtractorWIS(int noBits, int noTics, int selThresh);
        CV_WRAP BackgroundSubtractorWIS(int noBits, int noTics, double varThresh);
        CV_WRAP BackgroundSubtractorWIS(int noBits, int noTics, double varThresh, double incr, double decr);
        CV_WRAP BackgroundSubtractorWIS(Size frameSize, int frameType);
        CV_WRAP ~BackgroundSubtractorWIS();
        virtual void getBackgroundImage(Mat &img);
        //virtual void operator ()(InputArray _frame, OutputArray _fgmask);
        virtual void operator() ( const Mat& image, Mat& fgmask );
        virtual void initialize(Size frameSize, int frameType);
        virtual void printinfo(int size);
        virtual AlgorithmInfo* info() const;
        
    private:
        wvalue_t mentalDiscr(wentry_t **discr, wvalue_t *mentals);
        cache_entry_t *makeTupleCached(cache_entry_t *cache, unsigned char R, unsigned char G, unsigned char B, pix_t **neigh_map);
        void updateMaxColor(unsigned char *color, keyval_t *keys,unsigned char *NB, unsigned char *NG,unsigned char *NR);
        wisard_t *_wiznet = (wisard_t *)NULL;
        pix_t **_neigh_map;
        std::pair <int, int > **_neigh_map_raw;
        Mat _fgmask, _fgdetect;
        int histoSize;
        unsigned char **maxcolor;
        keyval_t **maxkeys;
        wvalue_t **_mentals;
        int *histoArray;
    };
}