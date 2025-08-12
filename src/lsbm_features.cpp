// src/lsbm_features.cpp
#include "lsbm_features.hpp"
#include <cmath>
#include <stdexcept>
#include <numeric>

using namespace cv;

namespace lsbm {

// --- Utilities ------------------------------------------------------------

static Mat to_gray_if_needed(const Mat &img) {
    if (img.channels() == 1) return img;
    Mat gray;
    cvtColor(img, gray, COLOR_RGB2GRAY);
    return gray;
}

// extract bit plane 0 or 1 etc from a single-channel 8-bit Mat
static Mat bit_plane(const Mat &channel, int bit) {
    CV_Assert(channel.type() == CV_8UC1);
    Mat plane(channel.size(), CV_8UC1);
    for (int y = 0; y < channel.rows; ++y) {
        const uchar* row = channel.ptr<uchar>(y);
        uchar* prow = plane.ptr<uchar>(y);
        for (int x = 0; x < channel.cols; ++x) {
            prow[x] = (row[x] >> bit) & 1;
        }
    }
    return plane;
}

// convert uchar binary mat to double vector and compute Pearson correlation
static double correlation(const Mat &A_u8, const Mat &B_u8) {
    Mat A, B;
    A_u8.convertTo(A, CV_64F);
    B_u8.convertTo(B, CV_64F);
    Scalar meanA = mean(A);
    Scalar meanB = mean(B);
    Mat A0 = A - meanA[0];
    Mat B0 = B - meanB[0];
    double num = A0.dot(B0);
    double den = std::sqrt(A0.dot(A0) * B0.dot(B0));
    if (den == 0.0) return 0.0;
    return num / den;
}

// correlation between two 1D vectors (probability distributions etc)
static double corr_vec(const std::vector<double>& a, const std::vector<double>& b) {
    int n = std::min(a.size(), b.size());
    double meanA = 0, meanB = 0;
    for (int i=0;i<n;++i){ meanA += a[i]; meanB += b[i]; }
    meanA /= n; meanB /= n;
    double num=0, denA=0, denB=0;
    for (int i=0;i<n;++i){
        double aa = a[i]-meanA, bb = b[i]-meanB;
        num += aa*bb; denA += aa*aa; denB += bb*bb;
    }
    double den = std::sqrt(denA*denB);
    if (den==0) return 0.0;
    return num/den;
}

// compute histogram probability vector (normalized)
static std::vector<double> hist_prob(const Mat &channel, int bins=256) {
    CV_Assert(channel.type()==CV_8UC1);
    std::vector<int> h(bins,0);
    for (int y=0;y<channel.rows;++y){
        const uchar* row = channel.ptr<uchar>(y);
        for (int x=0;x<channel.cols;++x) h[row[x]]++;
    }
    int total = channel.rows * channel.cols;
    std::vector<double> p(bins);
    for (int i=0;i<bins;++i) p[i] = double(h[i]) / double(total);
    return p;
}

// shift matrix by (k,l) (k rows, l cols) for correlation C(k,l) as in paper
static Mat shift_region(const Mat &M, int k, int l) {
    // create same-size matrix padded by zeros; but for correlation they compare overlapping blocks:
    // We'll return the overlapped-submatrix that matches the paper formula: return top-left part aligned.
    int rows = M.rows, cols = M.cols;
    int r0 = std::max(0, k), c0 = std::max(0, l);
    int r1 = std::min(rows, rows + k);
    int c1 = std::min(cols, cols + l);
    // We'll produce a cropped matrix for the overlapping region
    int h = rows - abs(k);
    int w = cols - abs(l);
    Mat out(h, w, M.type());
    for (int y=0;y<h;++y) {
        for (int x=0;x<w;++x) {
            int ys = (k>=0) ? y+k : y;
            int xs = (l>=0) ? x+l : x;
            out.at<uchar>(y,x) = M.at<uchar>(ys, xs);
        }
    }
    return out;
}

// compute autocorrelation C(k,l) for binary matrix M (uchar 0/1)
static double autocorr_shift(const Mat &M, int k, int l) {
    // produce two overlapped blocks X and shifted X per paper
    int rows = M.rows, cols = M.cols;
    int h = rows - abs(k);
    int w = cols - abs(l);
    if (h <= 0 || w <= 0) return 0.0;
    Mat A(h,w,CV_8UC1), B(h,w,CV_8UC1);
    for (int y=0;y<h;++y){
        for (int x=0;x<w;++x){
            int y1 = (k>=0) ? y : y - k;
            int x1 = (l>=0) ? x : x - l;
            int y2 = y1 + ((k>=0) ? -k : k);
            int x2 = x1 + ((l>=0) ? -l : l);
            // simpler approach: follow paper's X definitions: X(m-k:n-l) etc.
            // We'll compute A = M(y1+abs(min(0,k)), x1+abs(min(0,l))) and B = M(y1+abs(max(0,k)), x1+abs(max(0,l)))
            // but above is complicated; easier: build using direct indexing used in shift_region
            A.at<uchar>(y,x) = M.at<uchar>( (k>=0)? y : y - k, (l>=0)? x : x - l );
            B.at<uchar>(y,x) = M.at<uchar>( (k>=0)? y + k : y, (l>=0)? x + l : x );
        }
    }
    return correlation(A,B);
}

// ---------------- Haar one-level forward & inverse (float) -----------------
// forward: input single-channel float Mat, output LL, LH, HL, HH (each half size)
static void haar_forward_onelevel(const Mat &in, Mat &LL, Mat &LH, Mat &HL, Mat &HH) {
    // assume in is CV_32F
    int rows = in.rows, cols = in.cols;
    int r2 = rows/2, c2 = cols/2;
    LL = Mat::zeros(r2,c2,CV_32F);
    LH = Mat::zeros(r2,c2,CV_32F);
    HL = Mat::zeros(r2,c2,CV_32F);
    HH = Mat::zeros(r2,c2,CV_32F);

    // row transform (pairwise)
    Mat temp(rows, c2*2, CV_32F); // we'll store averages then diffs horizontally
    for (int y=0;y<rows;++y){
        for (int x=0, j=0; x+1<cols; x+=2, ++j){
            float a = in.at<float>(y,x), b = in.at<float>(y,x+1);
            temp.at<float>(y,j) = (a + b) / 2.0f;    // approx
            temp.at<float>(y,j + c2) = (a - b) / 2.0f; // detail
        }
    }
    // column transform on temp
    for (int x=0;x<c2;++x){
        for (int y=0, i=0; y+1<rows; y+=2, ++i){
            float a = temp.at<float>(y,x), b = temp.at<float>(y+1,x);
            LL.at<float>(i,x) = (a + b) / 2.0f;
            HL.at<float>(i,x) = (a - b) / 2.0f;
        }
        for (int y=0, i=0; y+1<rows; y+=2, ++i){
            float a = temp.at<float>(y,x+c2), b = temp.at<float>(y+1,x+c2);
            LH.at<float>(i,x) = (a + b) / 2.0f;
            HH.at<float>(i,x) = (a - b) / 2.0f;
        }
    }
}

// inverse one-level Haar from LL,LH,HL,HH -> reconstruct image float
static Mat haar_inverse_onelevel(const Mat &LL, const Mat &LH, const Mat &HL, const Mat &HH) {
    int r2 = LL.rows, c2 = LL.cols;
    int rows = r2*2, cols = c2*2;
    Mat temp(rows, c2*2, CV_32F);
    // inverse column transform to produce temp
    for (int x=0; x<c2; ++x){
        for (int i=0, y=0; i<r2; ++i, y+=2){
            float a = LL.at<float>(i,x), b = HL.at<float>(i,x);
            temp.at<float>(y, x) = (a + b);
            temp.at<float>(y+1, x) = (a - b);
        }
        for (int i=0, y=0; i<r2; ++i, y+=2){
            float a = LH.at<float>(i,x), b = HH.at<float>(i,x);
            temp.at<float>(y, x+c2) = (a + b);
            temp.at<float>(y+1, x+c2) = (a - b);
        }
    }
    // inverse row transform on temp -> out
    Mat out(rows, cols, CV_32F);
    for (int y=0; y<rows; ++y){
        for (int j=0, x=0; j<c2; ++j, x+=2){
            float approx = temp.at<float>(y,j);
            float detail = temp.at<float>(y,j + c2);
            out.at<float>(y,x) = approx + detail;
            out.at<float>(y,x+1) = approx - detail;
        }
    }
    return out;
}

// denoise single-channel 8U image using 1-level Haar with threshold t applied to HL/LH/HH absolute values
static Mat denoise_haar_threshold(const Mat &chan8u, double t) {
    Mat f; chan8u.convertTo(f, CV_32F);
    // ensure even dims
    int rows = f.rows - (f.rows % 2);
    int cols = f.cols - (f.cols % 2);
    Mat fcrop = f(Rect(0,0,cols,rows)).clone();
    Mat LL,LH,HL,HH;
    haar_forward_onelevel(fcrop, LL,LH,HL,HH);
    // zero small coefficients in HL,LH,HH absolute < t
    for (int y=0;y<HL.rows;++y) for (int x=0;x<HL.cols;++x) if (std::abs(HL.at<float>(y,x)) < t) HL.at<float>(y,x)=0;
    for (int y=0;y<LH.rows;++y) for (int x=0;x<LH.cols;++x) if (std::abs(LH.at<float>(y,x)) < t) LH.at<float>(y,x)=0;
    for (int y=0;y<HH.rows;++y) for (int x=0;x<HH.cols;++x) if (std::abs(HH.at<float>(y,x)) < t) HH.at<float>(y,x)=0;
    Mat rec = haar_inverse_onelevel(LL,LH,HL,HH);
    Mat out;
    rec.convertTo(out, CV_8U, 1.0, 0.0);
    // pad to original size by copying rows/cols beyond even crop unchanged from original
    Mat result = chan8u.clone();
    rec.convertTo(result(Rect(0,0,rec.cols, rec.rows)), CV_8U);
    return result;
}

// CE(t;k,l) correlation of E_t between overlapping blocks with shift (k,l)
static double CE_tkl(const Mat &chan8u, double t, int k, int l) {
    Mat den = denoise_haar_threshold(chan8u, t);
    Mat E = Mat::zeros(chan8u.size(), CV_8UC1);
    for (int y=0;y<chan8u.rows;++y){
        for (int x=0;x<chan8u.cols;++x) E.at<uchar>(y,x) = static_cast<uchar>( (int)chan8u.at<uchar>(y,x) - (int)den.at<uchar>(y,x) + 128 ); 
        // +128 to keep values non-negative; correlation depends on relative variation so OK
    }
    return autocorr_shift(E, k, l);
}

// ---- compute per-channel 41 features as C1..C41 ---------------------------
static std::vector<double> compute_channel_features(const Mat &chan) {
    // chan is CV_8UC1
    std::vector<double> feat;
    // C1 = cor(LSBP, LSBP2)
    Mat lsbp = bit_plane(chan, 0);
    Mat lsbp2 = bit_plane(chan, 1);
    feat.push_back(correlation(lsbp, lsbp2)); // C1

    // C2..C15 : autocorr with (k,l) pairs specified in paper
    std::vector<std::pair<int,int>> shifts = {
        {1,0},{2,0},{3,0},{4,0},{0,1},{0,2},{0,3},{0,4},
        {1,1},{2,2},{3,3},{4,4},{1,2},{2,1}
    };
    for (auto &p : shifts) feat.push_back( autocorr_shift(lsbp, p.first, p.second) );

    // C16: cor(He, Ho) where He are histogram even bins, Ho odd bins
    auto H = hist_prob(chan, 256);
    std::vector<double> He, Ho;
    for (int i=0;i<256;i+=2) He.push_back(H[i]);
    for (int i=1;i<256;i+=2) Ho.push_back(H[i]);
    feat.push_back( corr_vec(He, Ho) ); // C16

    // C17..C20: CH(1..4) where CH(l) = cor( Hl1, Hl2 ) definitions:
    // Hl1 = (rho0..rhoN-1-l), Hl2=(rho_l..)
    for (int l=1;l<=4;++l) {
        std::vector<double> Hl1, Hl2;
        for (int i=0;i<256-l;++i) Hl1.push_back(H[i]);
        for (int i=l;i<256;++i) Hl2.push_back(H[i]);
        feat.push_back( corr_vec(Hl1, Hl2) );
    }

    // C21..C41 : CE(t;k,l) for t in {1.5, 2, 2.5} and (k,l) set
    std::vector<double> tvals = {1.5, 2.0, 2.5};
    std::vector<std::pair<int,int>> klset = {
        {0,1},{1,0},{1,1},{0,2},{2,0},{1,2},{2,1}
    };
    for (double t : tvals) {
        for (auto &p : klset) {
            feat.push_back( CE_tkl(chan, t, p.first, p.second) );
        }
    }

    // Should be exactly 41 features now
    return feat;
}

// --------------- public functions -----------------------------------------

std::vector<double> extract_all_features(const Mat &image) {
    if (image.empty()) throw std::runtime_error("Image is empty");
    Mat img;
    if (image.type() == CV_8UC3) img = image.clone();
    else if (image.type() == CV_8UC1) cvtColor(image, img, COLOR_GRAY2RGB);
    else throw std::runtime_error("Unsupported image type; expect 8-bit RGB or gray");

    // OpenCV default is BGR for imread; here we assume bindings convert to RGB before calling.
    std::vector<Mat> channels(3);
    split(img, channels); // channels[0]=R if already converted to RGB

    // compute per-channel 41 features
    std::vector<double> all;
    for (int c=0;c<3;++c) {
        auto ch_feats = compute_channel_features(channels[c]);
        all.insert(all.end(), ch_feats.begin(), ch_feats.end());
    }

    // Crg, Crb, Cgb : abs(cor(Mr1, Mg1)) etc
    Mat Mr1 = bit_plane(channels[0], 0);
    Mat Mg1 = bit_plane(channels[1], 0);
    Mat Mb1 = bit_plane(channels[2], 0);
    all.push_back( std::abs( correlation(Mr1, Mg1) ) ); // Crg
    all.push_back( std::abs( correlation(Mr1, Mb1) ) ); // Crb
    all.push_back( std::abs( correlation(Mg1, Mb1) ) ); // Cgb

    // rgE, rbE, gbE for t in {1, 1.5, 2} (3 t values * 3 pairs = 9 features)
    std::vector<double> tvals = {1.0, 1.5, 2.0};
    for (double t : tvals) {
        // compute E_t per channel
        Mat Er = Mat::zeros(channels[0].size(), CV_8UC1);
        Mat Eg = Mat::zeros(channels[1].size(), CV_8UC1);
        Mat Eb = Mat::zeros(channels[2].size(), CV_8UC1);
        Mat dr = denoise_haar_threshold(channels[0], t);
        Mat dg = denoise_haar_threshold(channels[1], t);
        Mat db = denoise_haar_threshold(channels[2], t);
        for (int y=0;y<channels[0].rows;++y){
            for (int x=0;x<channels[0].cols;++x){
                int v1 = (int)channels[0].at<uchar>(y,x) - (int)dr.at<uchar>(y,x) + 128;
                int v2 = (int)channels[1].at<uchar>(y,x) - (int)dg.at<uchar>(y,x) + 128;
                int v3 = (int)channels[2].at<uchar>(y,x) - (int)db.at<uchar>(y,x) + 128;
                Er.at<uchar>(y,x) = (uchar)std::clamp(v1, 0, 255);
                Eg.at<uchar>(y,x) = (uchar)std::clamp(v2, 0, 255);
                Eb.at<uchar>(y,x) = (uchar)std::clamp(v3, 0, 255);
            }
        }
        all.push_back( correlation(Er, Eg) ); // rgE(t)
        all.push_back( correlation(Er, Eb) ); // rbE(t)
        all.push_back( correlation(Eg, Eb) ); // gbE(t)
    }

    // Count check: should be 41*3 + 3 + 9 = 135
    return all;
}

// selected features per paper -> 54 features
std::vector<double> extract_selected_features(const Mat &image) {
    // We implement exactly the subset described in the paper:
    // Per-channel: C1, C2, C6, C10, C14, C15, C16, C17, plus CE(t in {2.5,3} ; (0,1),(1,0),(1,1))
    // That gives 14 features per channel => 42. Then add rg/rb/gb E-correlations for t in {1,1.5,2} (3x3=9)
    // and Crg, Crb, Cgb (3) => total 54.

    if (image.empty()) throw std::runtime_error("Image is empty");
    Mat img;
    if (image.type() == CV_8UC3) img = image.clone();
    else if (image.type() == CV_8UC1) cvtColor(image, img, COLOR_GRAY2RGB);
    else throw std::runtime_error("Unsupported image type");

    std::vector<Mat> channels(3);
    split(img, channels);

    std::vector<double> out;

    // per-channel features
    for (int c=0;c<3;++c) {
        Mat chan = channels[c];
        // compute base stats needed
        Mat lsbp = bit_plane(chan,0);
        Mat lsbp2 = bit_plane(chan,1);
        out.push_back( correlation(lsbp, lsbp2) ); // C1
        out.push_back( autocorr_shift(lsbp,1,0) ); // C2
        out.push_back( autocorr_shift(lsbp,0,1) ); // C6
        out.push_back( autocorr_shift(lsbp,1,1) ); // C10
        out.push_back( autocorr_shift(lsbp,2,1) ); // C14 (note: mapping from paper) 
        out.push_back( autocorr_shift(lsbp,2,0) ); // C15 (some ordering - preserve paper mapping)
        // C16 cor(He,Ho)
        auto H = hist_prob(chan,256);
        std::vector<double> He, Ho;
        for (int i=0;i<256;i+=2) He.push_back(H[i]);
        for (int i=1;i<256;i+=2) Ho.push_back(H[i]);
        out.push_back( corr_vec(He,Ho) ); // C16
        // C17 (CH(1))
        {
            int l = 1;
            std::vector<double> Hl1, Hl2;
            for (int i=0;i<256-l;++i) Hl1.push_back(H[i]);
            for (int i=l;i<256;++i) Hl2.push_back(H[i]);
            out.push_back( corr_vec(Hl1, Hl2) ); // C17
        }
        // CE features for t in {2.5, 3.0} and shifts (0,1),(1,0),(1,1)
        std::vector<double> tvals = {2.5, 3.0};
        std::vector<std::pair<int,int>> kl = {{0,1},{1,0},{1,1}};
        for (double t : tvals) {
            for (auto &p : kl) {
                out.push_back( CE_tkl(chan, t, p.first, p.second) );
            }
        }
        // That makes 8 + 6 = 14 per channel
    }

    // Crg, Crb, Cgb absolute LSBP correlations
    Mat Mr1 = bit_plane(channels[0],0);
    Mat Mg1 = bit_plane(channels[1],0);
    Mat Mb1 = bit_plane(channels[2],0);
    out.push_back( std::abs(correlation(Mr1, Mg1)) );
    out.push_back( std::abs(correlation(Mr1, Mb1)) );
    out.push_back( std::abs(correlation(Mg1, Mb1)) );

    // rg/rb/gb E correlations for t in {1,1.5,2}
    std::vector<double> tvals = {1.0, 1.5, 2.0};
    for (double t : tvals) {
        Mat dr = denoise_haar_threshold(channels[0], t);
        Mat dg = denoise_haar_threshold(channels[1], t);
        Mat db = denoise_haar_threshold(channels[2], t);
        Mat Er(channels[0].size(), CV_8UC1), Eg=Er, Eb=Er;
        for (int y=0;y<channels[0].rows;++y){
            for (int x=0;x<channels[0].cols;++x){
                int v1 = (int)channels[0].at<uchar>(y,x) - (int)dr.at<uchar>(y,x) + 128;
                int v2 = (int)channels[1].at<uchar>(y,x) - (int)dg.at<uchar>(y,x) + 128;
                int v3 = (int)channels[2].at<uchar>(y,x) - (int)db.at<uchar>(y,x) + 128;
                Er.at<uchar>(y,x) = (uchar)std::clamp(v1, 0, 255);
                Eg.at<uchar>(y,x) = (uchar)std::clamp(v2, 0, 255);
                Eb.at<uchar>(y,x) = (uchar)std::clamp(v3, 0, 255);
            }
        }
        out.push_back( correlation(Er, Eg) );
        out.push_back( correlation(Er, Eb) );
        out.push_back( correlation(Eg, Eb) );
    }

    // total should be 54
    return out;
}

} // namespace lsbm

