//
// Change Detection based on Wisard 1.0
//
// Created by Maurizio Giordano
// Copyright 2015
//

#include "bgwis.hpp"
// C++ headers
#include <iostream>
#include <iomanip>
#include <string>

// OpenCv include
#include <opencv2/core/core.hpp>

#define RESET   "\033[0m"

//#include "wnet_lib.hpp"

using namespace cv;
using namespace std;

namespace cv {
    
    wvalue_t BackgroundSubtractorWIS::mentalDiscr(wentry_t **discr, wvalue_t *mentals) {
        wentry_t *p, *m;
        int offset=0, b;
        wvalue_t maxvalue=0, value;
        for (int i=0; i< noTics * 3; i++) mentals[i] = 0;
        for (int neuron=0,offset=0;neuron<noRams;neuron++,offset+=noBits) {
            m = discr[neuron];
            for(p=m;;p=p->next) {
                if (p->next==m) break;
                for (b=0;b<noBits;b++) {
                    if (((p->next->key)>>(wkey_t)(noBits - 1 - b) & 1) > 0) {
                        //value = mentals[_neigh_map_raw[offset + b]] += p->next->value;
                        if (maxvalue < value) maxvalue = value;
                    }
                }
            }
        }
        return maxvalue;
    }
    // Make tuple with Cache support
    cache_entry_t *BackgroundSubtractorWIS::makeTupleCached(cache_entry_t *cache, unsigned char R, unsigned char G, unsigned char B, pix_t **neigh_map) {
        cache_entry_t *p, *prec;
        int cr = (int)(R / dimTics);
        int cg = (int)(G / dimTics);
        int cb = (int)(B / dimTics);
        register int neuron, k;
        int tmp, *ptr;
        // scan cache for hit or miss
        p = cache;
        prec = cache->prev;
        tcount ++;
        for (;;) {
            if (p->cr == cr && p->cg == cg && p->cb == cb) {  // cache hit (move found in front)
                //printf("HIT:  ");
                hits++;
                if (p == cache) {  // hit and in front
                    cache->weight += cache->idx;
                    cache->idx++;
                } else {            // hit and not in front (move to front)
                    prec->next = p->next;   // remove item
                    p->next->prev=prec;
                    cache->prev->next = p;
                    p->prev = cache->prev;
                    p->next = cache;
                    cache->prev = p;
                    p->idx = 1;
                    cache = p;
                }
                return cache;
            }
            if (p->next == cache) {
                // move top on first non-empty
                //printf("MISS: ");
                misses++;
                cache = cache->prev;
                cache->cr = cr;
                cache->cg = cg;
                cache->cb = cb;
                // initialize tuple
                for (neuron=0;neuron<noRams;neuron++) cache->tuple[neuron] = 0;
                // compute tuple
                for (k=0;k<noTics;k++) {
                    if (k<cr) cache->tuple[neigh_map[0][k].first] |= neigh_map[0][k].second;
                    if (k<cg) cache->tuple[neigh_map[1][k].first] |= neigh_map[1][k].second;
                    if (k<cb) cache->tuple[neigh_map[2][k].first] |= neigh_map[2][k].second;
                }
                cache->weight = 0;
                cache->idx = 1;
                return cache;
            }
            p = p->next;
            prec = prec->next;
        }
    }
    
    BackgroundSubtractorWIS::~BackgroundSubtractorWIS()
    {
    }
    
    BackgroundSubtractorWIS::BackgroundSubtractorWIS() {
        noBits = defaultNoBits;
        noTics = defaultNoTics;
        dimTics = (int) (256 / noTics);
        trainIncr = defaultTrainIncr;
        trainDecr = defaultTrainDecr;
        selectThreshold = defaultSelectThreshold;
        varThreshold = defaultVarThreshold;
        varWatermark = defaultVarWatermark;
        varUpWatermark = defaultVarUpWatermark;
        cacheSize = defaultCacheSize;
        learningStage = defaultLearningStage;
        hits = misses = tcount = 0;
    }
    
    BackgroundSubtractorWIS::BackgroundSubtractorWIS(int nb, int nt) {
        noBits = nb;
        noTics = nt;
        trainIncr = defaultTrainIncr;
        trainDecr = defaultTrainDecr;
        selectThreshold = defaultSelectThreshold;
        varThreshold = defaultVarThreshold;
        varWatermark = defaultVarWatermark;
        varUpWatermark = defaultVarUpWatermark;
        cacheSize = defaultCacheSize;
        learningStage = defaultLearningStage;
        hits = misses = tcount = 0;
    }
    
    BackgroundSubtractorWIS::BackgroundSubtractorWIS(int nb, int nt, int st) {
        noBits = nb;
        noTics = nt;
        selectThreshold = st;
        trainIncr = defaultTrainIncr;
        trainDecr = defaultTrainDecr;
        varThreshold = defaultVarThreshold;
        varWatermark = defaultVarWatermark;
        varUpWatermark = defaultVarUpWatermark;
        cacheSize = defaultCacheSize;
        hits = misses = tcount = 0;
    }
    
    BackgroundSubtractorWIS::BackgroundSubtractorWIS(int nb, int nt, double th) {
        noBits = nb;
        noTics = nt;
        selectThreshold = defaultSelectThreshold;
        trainIncr = defaultTrainIncr;
        trainDecr = defaultTrainDecr;
        varThreshold = th;
        varWatermark = defaultVarWatermark;
        varUpWatermark = defaultVarUpWatermark;
        cacheSize = defaultCacheSize;
        hits = misses = tcount = 0;
    }

    BackgroundSubtractorWIS::BackgroundSubtractorWIS(int nb, int nt, double th, double incr, double decr) {
        noBits = nb;
        noTics = nt;
        selectThreshold = defaultSelectThreshold;
        trainIncr = incr;
        trainDecr = decr;
        varThreshold = th;
        varWatermark = defaultVarWatermark;
        varUpWatermark = defaultVarUpWatermark;
        cacheSize = defaultCacheSize;
        hits = misses = tcount = 0;
    }

    BackgroundSubtractorWIS::BackgroundSubtractorWIS(Size _frameSize, int _frameType) {
        frameSize = _frameSize;
        frameType = _frameType;
        frameNum_ = 0;
        string colormode = "RGB";
        
        int nchannels = CV_MAT_CN(frameType);
        CV_Assert( nchannels <= CV_CN_MAX );
        _wiznet = net_create(noBits,frameSize.width, frameSize.height,colormode, noTics, cacheSize);
        noRams = _wiznet->n_ram;
        histoSize = frameSize.width * frameSize.height * (noTics + 2) * nchannels;
        histoArray = (int *)malloc(sizeof(int) * histoSize);
        for (int i=0; i < histoSize; i++) histoArray[i]=0;
        maxkeys = (keyval_t **)malloc(sizeof(keyval_t *) * frameSize.width * frameSize.height);
        maxcolor = (unsigned char **)malloc(sizeof(unsigned char *) * frameSize.width * frameSize.height);
        for (int i=0; i < frameSize.width * frameSize.height; i++) {
            maxkeys[i] = (keyval_t *)malloc(sizeof(keyval_t) * noRams);
            maxcolor[i] = (unsigned char *)malloc(sizeof(unsigned char) * noTics * 3);
            for (int x=0; x < noTics * 3; x++) maxcolor[i][x] = (unsigned char)0;
        }
        _neigh_map = _wiznet->neigh_map;
        
        // Create mask frame
        _fgmask.create( frameSize, CV_8U );
        _fgdetect.create( frameSize, frameType );
    }
    
    void BackgroundSubtractorWIS::initialize(Size _frameSize, int _frameType) {
        frameSize = _frameSize;
        frameType = _frameType;
        frameNum_ = 0;
        string colormode = "RGB";
        
        int nchannels = CV_MAT_CN(frameType);
        dimTics = (int) (256 / noTics);
        CV_Assert( nchannels <= CV_CN_MAX );
        _wiznet = net_create(noBits,frameSize.width, frameSize.height,colormode, noTics, cacheSize);
        noRams = _wiznet->n_ram;
        histoSize = frameSize.width * frameSize.height * (noTics + 2) * nchannels;
        histoArray = (int *)malloc(sizeof(int) * histoSize);
        for (int i=0; i < histoSize; i++) histoArray[i]=0;
        maxkeys = (keyval_t **)malloc(sizeof(keyval_t *) * frameSize.width * frameSize.height);
        maxcolor = (unsigned char **)malloc(sizeof(unsigned char *) * frameSize.width * frameSize.height);
        for (int i=0; i < frameSize.width * frameSize.height; i++) {
            maxkeys[i] = (keyval_t *)malloc(sizeof(keyval_t) * noRams);
            maxcolor[i] = (unsigned char *)malloc(sizeof(unsigned char) * noTics * 3);
            for (int x=0; x < noTics * 3; x++) maxcolor[i][x] = (unsigned char)0;
        }
        _neigh_map = _wiznet->neigh_map;
        
        // Create mask frame
        _fgmask.create( frameSize, CV_8U );
        _fgdetect.create( frameSize, _frameType );
        _fgdetect = Scalar(0, 0, 0);
    }
    
    void BackgroundSubtractorWIS::operator ()( const Mat& image, Mat& fgmask) {
        const uchar *data;
        uchar *odata, *fgdata, *color;
        wentry_t **discr;
        wkey_t *tuple;
        keyval_t *keys;
        wvalue_t *mentals;
        int cr, cg, cb;
        cache_entry_t * cache;
        pix_t **neigh_map;
        uchar B,G,R;
        wisard_t *wiznet = _wiznet;
        int *histos = histoArray;
        
        fgmask = _fgmask;
        Mat fgdetect = _fgdetect;
        fgdetect = Scalar(0, 0, 0);
        double sum=0;
        neigh_map = wiznet->neigh_map;
#pragma omp parallel for schedule(static) shared(image,fgmask,fgdetect,wiznet,neigh_map) private(sum,discr,cache,data,odata,fgdata,R,G,B,color,keys)
        for (int j=0; j<image.rows; j++) {
            data= image.ptr<uchar>(j);
            odata= fgmask.ptr<uchar>(j);
            fgdata= fgdetect.ptr<uchar>(j);
            for (int i=0; i<image.cols; i++) {
                discr = wiznet->net[j*image.cols + i]; // get discriminator of pixel
                cache = wiznet->cachearray[j*image.cols + i]; // get cache for pixel
                // make tuple
                //B = *data++; G = *data++; R = *data++;
                R = *data++; G = *data++; B = *data++;
               	cache = makeTupleCached(cache,R,G,B,neigh_map);
                wiznet->cachearray[j*image.cols + i] = cache;
                color = maxcolor[j*image.cols + i];
                keys = maxkeys[j*image.cols + i];
                // classify
                sum = 0;
                for (int neuron=0;neuron<noRams;neuron++) {
                    //if (wram_get(discr[neuron],cache->tuple[neuron]) > varWatermark) {
                    //    sum++;
                    //}
                    if (learningStage > 0 || cache->weight > selectThreshold) keys[neuron] = wram_up_key_down_rest(discr[neuron], cache->tuple[neuron],trainIncr,trainDecr,varUpWatermark);
                }
 
                // update output mask
                //if (sum/(double)noRams >= varThreshold) {  // (sigma) Pixel is black (BG)
                //    *odata++ = (uchar)0; // set pixel to BG
                //} else {                                   // pixels is white (FG)
                //    *odata++ = (uchar)255; // set pixel to FG
                //}
                updateMaxColor(color,keys,fgdata++,fgdata++,fgdata++);  // update bgmodel in all pixels                
            }
        }
        if (learningStage > 0) {
            learningStage--;
        }
    }
    
    void BackgroundSubtractorWIS::updateMaxColor(unsigned char *color, keyval_t *keys, unsigned char *NB, unsigned char *NG,unsigned char *NR) {
        unsigned char *B, *G, *R;
        B = color;
        G = color+noTics;
        R = color+noTics+noTics;
        //int cb=noTics,cg=noTics,cr=noTics;
        int cb=0,cg=0,cr=0;
        for (int k=0; k < noTics; k++) {
            *B = (unsigned char)((keys[_neigh_map[0][k].first].first & _neigh_map[0][k].second) / _neigh_map[0][k].second);
            *G = (unsigned char)((keys[_neigh_map[1][k].first].first & _neigh_map[1][k].second) / _neigh_map[1][k].second);
            *R = (unsigned char)((keys[_neigh_map[2][k].first].first & _neigh_map[2][k].second) / _neigh_map[2][k].second);
            //if (k < cb && *B == (unsigned char)0 ) { cb = k; }
            //if (k < cg && *G == (unsigned char)0 ) { cg = k; }
            //if (k < cr && *R == (unsigned char)0 ) { cr = k; }
            if (*B == (unsigned char)1 ) { cb++; }
            if (*G == (unsigned char)1 ) { cg++; }
            if (*R == (unsigned char)1 ) { cr++; }
            B++; G++; R++;
        }
        *NB = (unsigned char)(cb * dimTics);
        *NG = (unsigned char)(cg * dimTics);
        *NR = (unsigned char)(cr * dimTics);
    }
    
    void BackgroundSubtractorWIS::getBackgroundImage(Mat &img) {
        img = _fgdetect;
    }
    
    void BackgroundSubtractorWIS::printinfo(int fldsize) {
        cout << left << setw(fldsize) << setfill(' ') << "noBits: " << noBits << endl;
        cout << left << setw(fldsize) << setfill(' ') << "noTics: " << noTics << "(" << dimTics << ")" << endl;
        cout << left << setw(fldsize) << setfill(' ') << "noRams: " << noRams <<  endl;
        cout << left << setw(fldsize) << setfill(' ') << "Train Policy: " << trainIncr << ":" << trainDecr <<  endl;
        cout << left << setw(fldsize) << setfill(' ') << "Classification Thresh: " << varThreshold << endl;
        cout << left << setw(fldsize) << setfill(' ') << "Selection Thresh: " << selectThreshold << endl;
        cout << left << setw(fldsize) << setfill(' ') << "Watermark: " << varWatermark << endl;
        cout << left << setw(fldsize) << setfill(' ') << "Uppermark: " << varUpWatermark << endl;
        cout << left << setw(fldsize) << setfill(' ') << "LearningStage: " << learningStage << endl;
        cout << left << setw(fldsize) << setfill(' ') << "Cachesize: " << cacheSize << endl;
        ++frameNum_;
    }
    
    CV_INIT_ALGORITHM(BackgroundSubtractorWIS, "WISARD BG Subtractor", // <-- keyword for Algorithm::create()
                      obj.info()->addParam(obj, "noBits", obj.noBits ,false,0,0,"number of bits");
                      obj.info()->addParam(obj, "noTics", obj.noTics ,false,0,0,"number of tics");
                      obj.info()->addParam(obj, "trainIncr", obj.trainIncr ,false,0,0,"forget step");
                      obj.info()->addParam(obj, "trainDecr", obj.trainDecr ,false,0,0,"reinforce step");
                      obj.info()->addParam(obj, "selectThreshold", obj.selectThreshold ,false,0,0,"selection threshold");
                      obj.info()->addParam(obj, "varThreshold", obj.varThreshold ,false,0,0,"classification threshold");
                      obj.info()->addParam(obj, "varWatermark", obj.varWatermark ,false,0,0,"down water mark");
                      obj.info()->addParam(obj, "varUpWatermark", obj.varUpWatermark ,false,0,0,"up water mark");
                      obj.info()->addParam(obj, "learningStage", obj.learningStage ,false,0,0,"training stage");
                      obj.info()->addParam(obj, "cacheSize", obj.cacheSize ,false,0,0,"chace size for tuples")
                      );
    
}