#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <vitis/ai/facedetect.hpp>
#include <map>
#include <vector>
#include <cmath>

// -----------------------------------------------------------------------------
// Utility: IoU (Intersection over Union) between two rectangles
// Returns 0.0 (no overlap) to 1.0 (identical boxes)
// -----------------------------------------------------------------------------
float iou(const cv::Rect& a, const cv::Rect& b) {
    int ix  = std::max(a.x, b.x);
    int iy  = std::max(a.y, b.y);
    int ix2 = std::min(a.x + a.width,  b.x + b.width);
    int iy2 = std::min(a.y + a.height, b.y + b.height);
    if (ix2 <= ix || iy2 <= iy) return 0.0f;
    float inter  = (float)(ix2 - ix) * (iy2 - iy);
    float unionA = a.width * a.height;
    float unionB = b.width * b.height;
    return inter / (unionA + unionB - inter);
}

// -----------------------------------------------------------------------------
// Smooth a bounding box using Exponential Moving Average (EMA)
// alpha=0.9 → snappy/jittery   alpha=0.6 → balanced   alpha=0.3 → smooth/laggy
// -----------------------------------------------------------------------------
cv::Rect smoothRect(const cv::Rect& prev, const cv::Rect& curr, float alpha = 0.6f) {
    return cv::Rect(
        (int)(alpha * curr.x      + (1 - alpha) * prev.x),
        (int)(alpha * curr.y      + (1 - alpha) * prev.y),
        (int)(alpha * curr.width  + (1 - alpha) * prev.width),
        (int)(alpha * curr.height + (1 - alpha) * prev.height)
    );
}

// -----------------------------------------------------------------------------
// TrackedFace: one instance per active face ID
// -----------------------------------------------------------------------------
struct TrackedFace {
    cv::Rect   box;           // raw DPU detection this frame
    cv::Rect   smoothedBox;   // EMA-smoothed box (drawn on screen)
    cv::Point2f centroid;     // centre point (x, y)
    cv::Point2f velocity;     // pixel displacement since last frame
    int missing = 0;          // frames since last matched detection
    int age     = 0;          // total frames this ID has been alive
};

// -----------------------------------------------------------------------------
// FaceTracker: manages all active face IDs across frames
// -----------------------------------------------------------------------------
class FaceTracker {
public:
    int   maxMissing = 40;     // drop ID after this many unmatched frames
    float minIoU     = 0.25f;  // minimum IoU to consider a match
    float maxDist    = 150.0f; // maximum centroid distance (pixels) for a match
    int   minAge     = 3;      // minimum age before an ID label is displayed

    std::map<int, cv::Rect> update(const std::vector<cv::Rect>& detections) {
        // Step 1: increment missing counter for every tracked face
        for (auto& [id, tf] : tracked_) tf.missing++;

        if (!detections.empty()) {
            // Pre-compute detection centroids
            std::vector<cv::Point2f> dCentroids;
            for (auto& r : detections)
                dCentroids.push_back({r.x + r.width / 2.f, r.y + r.height / 2.f});

            std::vector<bool> detUsed(detections.size(), false);

            // Step 2: match each tracked face to the best detection
            for (auto& [id, tf] : tracked_) {
                float bestScore = -1.0f;
                int   bestIdx   = -1;

                // predict position using last known velocity
                cv::Rect predicted = tf.box;
                predicted.x += (int)tf.velocity.x;
                predicted.y += (int)tf.velocity.y;

                for (int i = 0; i < (int)detections.size(); i++) {
                    if (detUsed[i]) continue;
                    float iouScore  = iou(predicted, detections[i]);
                    float dist      = cv::norm(tf.centroid - dCentroids[i]);
                    float distScore = std::max(0.0f, 1.0f - dist / maxDist);
                    float score     = iouScore * 0.7f + distScore * 0.3f;

                    if ((iouScore >= minIoU || dist < maxDist * 0.5f) && score > bestScore) {
                        bestScore = score;
                        bestIdx   = i;
                    }
                }

                if (bestIdx >= 0) {
                    cv::Point2f newCentroid = dCentroids[bestIdx];
                    tf.velocity    = cv::Point2f(newCentroid.x - tf.centroid.x,
                                                 newCentroid.y - tf.centroid.y);
                    tf.centroid    = newCentroid;
                    tf.smoothedBox = smoothRect(tf.smoothedBox, detections[bestIdx]);
                    tf.box         = detections[bestIdx];
                    tf.missing     = 0;
                    tf.age++;
                    detUsed[bestIdx] = true;
                }
            }

            // Step 3: register unmatched detections as new face IDs
            for (int i = 0; i < (int)detections.size(); i++) {
                if (!detUsed[i]) {
                    TrackedFace tf;
                    tf.box = tf.smoothedBox = detections[i];
                    tf.centroid  = dCentroids[i];
                    tf.velocity  = {0, 0};
                    tf.missing   = 0;
                    tf.age       = 0;
                    tracked_[nextID_++] = tf;
                }
            }
        }

        // Step 4: decay velocity for missing faces
        for (auto& [id, tf] : tracked_) {
            if (tf.missing > 0) {
                tf.velocity      *= 0.8f;
                tf.smoothedBox.x += (int)tf.velocity.x;
                tf.smoothedBox.y += (int)tf.velocity.y;
            }
        }

        // Step 5: remove stale IDs; return only confirmed faces
        std::map<int, cv::Rect> result;
        for (auto it = tracked_.begin(); it != tracked_.end(); ) {
            if (it->second.missing > maxMissing) {
                it = tracked_.erase(it);
            } else {
                if (it->second.age >= minAge || it->second.missing == 0)
                    result[it->first] = it->second.smoothedBox;
                ++it;
            }
        }
        return result;
    }

private:
    std::map<int, TrackedFace> tracked_;
    int nextID_ = 1;
};

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_name> <camera_id>\n"
                  << "  e.g. ./test_video_facedetect_tracked densebox_320_320 0\n";
        return -1;
    }

    // Load face-detection model onto the DPU
    auto det     = vitis::ai::FaceDetect::create(argv[1], true);
    auto tracker = FaceTracker();

    // Open the camera (CAP_PROP_BUFFERSIZE=1 → always get the freshest frame)
    cv::VideoCapture cap(std::stoi(argv[2]));
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return -1; }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    static const cv::Scalar COLORS[] = {
        {0,255,0}, {0,200,255}, {255,100,0}, {180,0,255}, {0,255,180}, {255,255,0}
    };

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // --- Run DPU inference ---
        auto result = det->run(frame);

        // Convert normalised [0,1] coords to pixel coords; filter tiny boxes
        std::vector<cv::Rect> boxes;
        for (auto& r : result.rects) {
            int x = (int)(r.x * frame.cols),      y = (int)(r.y * frame.rows);
            int w = (int)(r.width * frame.cols),   h = (int)(r.height * frame.rows);
            x = std::max(0, x);  y = std::max(0, y);
            w = std::min(w, frame.cols - x);  h = std::min(h, frame.rows - y);
            if (w > 10 && h > 10) boxes.emplace_back(x, y, w, h);
        }

        // --- NMS: remove duplicate overlapping detections ---
        std::vector<cv::Rect> filtered;
        if (!boxes.empty()) {
            std::vector<int>   indices;
            std::vector<float> scores(boxes.size(), 1.0f);
            cv::dnn::NMSBoxes(boxes, scores, 0.3f, 0.35f, indices);
            for (int i : indices) filtered.push_back(boxes[i]);
        }

        // --- Update tracker and draw results ---
        auto tracked = tracker.update(filtered);

        for (auto& [id, box] : tracked) {
            cv::Rect safe = box & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe.width < 5 || safe.height < 5) continue;

            cv::Scalar color = COLORS[(id - 1) % 6];

            // Draw bounding box
            cv::rectangle(frame, safe, color, 2);

            // Corner accents
            int clen = safe.width / 6;
            cv::line(frame, safe.tl(), safe.tl() + cv::Point( clen,    0), color, 3);
            cv::line(frame, safe.tl(), safe.tl() + cv::Point(    0, clen), color, 3);
            cv::line(frame, safe.br(), safe.br() - cv::Point( clen,    0), color, 3);
            cv::line(frame, safe.br(), safe.br() - cv::Point(    0, clen), color, 3);

            // ID label with filled background
            std::string label = "Face #" + std::to_string(id);
            int baseLine = 0;
            auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.65, 2, &baseLine);
            int lx = safe.x, ly = std::max(sz.height + 8, safe.y - 4);
            cv::rectangle(frame,
                cv::Point(lx, ly - sz.height - 6),
                cv::Point(lx + sz.width + 6, ly),
                color, cv::FILLED);
            cv::putText(frame, label, cv::Point(lx + 3, ly - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2);
        }

        // Face count overlay
        std::string countText = "Faces: " + std::to_string(tracked.size());
        cv::putText(frame, countText, cv::Point(10, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);

        cv::imshow("Face Tracker", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
