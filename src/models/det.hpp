#pragma once
#include "base.hpp"
#include <opencv2/opencv.hpp>

class Det : public base_model
{
private:
    double contour_score(const cv::Mat &binary, const std::vector<cv::Point> &contour)
    {
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.x < 0)
            rect.x = 0;
        if (rect.y < 0)
            rect.y = 0;
        if (rect.x + rect.width > binary.cols)
            rect.width = binary.cols - rect.x;
        if (rect.y + rect.height > binary.rows)
            rect.height = binary.rows - rect.y;

        cv::Mat binROI = binary(rect);

        cv::Mat mask = cv::Mat::zeros(rect.height, rect.width, CV_8U);
        std::vector<cv::Point> roiContour;
        for (size_t i = 0; i < contour.size(); i++)
        {
            cv::Point pt = cv::Point(contour[i].x - rect.x, contour[i].y - rect.y);
            roiContour.push_back(pt);
        }

        std::vector<std::vector<cv::Point>> roiContours = {roiContour};
        cv::fillPoly(mask, roiContours, cv::Scalar(255));

        double score = cv::mean(binROI, mask).val[0];
        return score / 255.f;
    }

public:
    int inference(cv::Mat input, ax_ocr_result_t *result) override
    {
        // --- Letterbox 居中填充开始 ---
        cv::Mat padded; // 用于存储 letterbox 后的图像
        float scale_w;
        float scale_h;

        // 计算缩放比例
        float ratio_w = (float)input_width / input.cols;
        float ratio_h = (float)input_height / input.rows;
        float ratio = std::min(ratio_w, ratio_h);

        // 确定新的尺寸
        int new_width = (int)(input.cols * ratio);
        int new_height = (int)(input.rows * ratio);

        // 计算填充量
        int pad_w = input_width - new_width;
        int pad_h = input_height - new_height;

        // 缩放图像
        cv::Mat resized_img;
        cv::resize(input, resized_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

        // 计算居中填充的偏移量
        int top = pad_h / 2;
        int bottom = pad_h - top;
        int left = pad_w / 2;
        int right = pad_w - left;

        // 进行填充，使用灰色（114）填充背景，这是 YOLO 系列模型常用的填充值
        cv::copyMakeBorder(resized_img, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        // 记录实际缩放因子，用于后处理时将坐标映射回原始图像尺寸
        scale_w = (float)input.cols / new_width;  // 原始宽 / 缩放后宽
        scale_h = (float)input.rows / new_height; // 原始高 / 缩放后高

        // --- Letterbox 居中填充结束 ---

        float *input_data = (float *)m_runner->get_input(0).pVirAddr;
        // 使用 padded.data 作为输入图像数据
        unsigned char *img_data = (unsigned char *)padded.data;

        int letterbox_cols = padded.cols; // 等于 input_width
        int letterbox_rows = padded.rows; // 等于 input_height

        // 将 HWC (Height-Width-Channel) 转换为 CHW (Channel-Height-Width) 并转换为 float
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    // 注意：这里没有进行归一化 (0-1) 或减均值/除方差，只是简单地从 uint8 转换为 float
                    input_data[out_index] = float(img_data[in_index]);
                }
            }
        }

        m_runner->inference();

        // 模型输出处理
        cv::Mat pred = cv::Mat(m_runner->get_output(0).vShape[2], m_runner->get_output(0).vShape[3], CV_32FC1, m_runner->get_output(0).pVirAddr);

        pred.convertTo(pred, CV_8UC1, 255);

        cv::Mat bitmap;

        // 假设 input_width 和 input_height 是模型输入尺寸，pred 是模型输出的概率图，尺寸与 input_width/input_height 相同或成比例
        cv::threshold(pred, bitmap, threshold * 255, 255, cv::THRESH_BINARY);

        // boxes from bitmap
        {
            const float box_thresh = 0.6f;
            const float enlarge_ratio = 1.95f;

            const float min_size = 3;
            const int max_candidates = 1000;

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;

            cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            contours.resize(std::min(contours.size(), (size_t)max_candidates));

            result->num_objs = 0;

            // --- 坐标映射调整开始 ---
            // 原始代码中的 scale_w/scale_h 是 (原始宽/输入宽)，这适用于简单 resize。
            // Letterbox 需要考虑缩放比例和填充偏移。

            // 用于将 **模型输出 (pred/bitmap) 坐标** 映射到 **letterbox 后的 resized_img 坐标**
            // pred 的尺寸和 letterbox 后的 padded 图像尺寸 (input_width/input_height) 相同。
            float ratio_scale = ratio; // 图像缩放因子，用于将 bitmap 坐标 (input_width/input_height) 映射到原图的缩放后区域 (new_width/new_height)

            // letterbox 居中填充的偏移量
            float offset_x = (float)left;
            float offset_y = (float)top;

            // 原始图像尺寸
            float original_w = (float)input.cols;
            float original_h = (float)input.rows;

            // 最终的后处理缩放因子，将 **bitmap 坐标** 映射回 **原始图像坐标**
            float final_scale_w = original_w / (float)new_width;
            float final_scale_h = original_h / (float)new_height;

            // --- 坐标映射调整结束 ---

            for (size_t i = 0; i < contours.size(); i++)
            {
                const std::vector<cv::Point> &contour = contours[i];
                if (contour.size() <= 2)
                    continue;

                double score = contour_score(pred, contour);
                if (score < box_thresh)
                    continue;

                cv::RotatedRect rrect = cv::minAreaRect(contour);

                float rrect_maxwh = std::max(rrect.size.width, rrect.size.height);
                if (rrect_maxwh < min_size)
                    continue;

                // ... (省略角度/方向调整代码，与原代码相同) ...
                int orientation = 0;
                if (rrect.angle >= -30 && rrect.angle <= 30 && rrect.size.height > rrect.size.width * 2.7)
                {
                    // vertical text
                    orientation = 1;
                }
                if ((rrect.angle <= -60 || rrect.angle >= 60) && rrect.size.width > rrect.size.height * 2.7)
                {
                    // vertical text
                    orientation = 1;
                }

                if (rrect.angle < -30)
                {
                    // make orientation from -90 ~ -30 to 90 ~ 150
                    rrect.angle += 180;
                }
                if (orientation == 0 && rrect.angle < 30)
                {
                    // make it horizontal
                    rrect.angle += 90;
                    std::swap(rrect.size.width, rrect.size.height);
                }
                if (orientation == 1 && rrect.angle >= 60)
                {
                    // make it vertical
                    rrect.angle -= 90;
                    std::swap(rrect.size.width, rrect.size.height);
                }
                // enlarge
                rrect.size.height += rrect.size.width * (enlarge_ratio - 1);
                rrect.size.width *= enlarge_ratio;

                // --- 坐标映射回原始图像尺寸 ---

                // 1. 消除 letterbox 填充的影响 (将坐标从 padded 图像坐标系转到 resized_img 图像坐标系)
                rrect.center.x = rrect.center.x - offset_x;
                rrect.center.y = rrect.center.y - offset_y;

                // 2. 将坐标从 resized_img 图像坐标系映射到原始图像坐标系
                // 尺寸也需要根据缩放因子进行调整
                rrect.center.x = rrect.center.x * final_scale_w;
                rrect.center.y = rrect.center.y * final_scale_h;
                rrect.size.width = rrect.size.width * final_scale_w;
                rrect.size.height = rrect.size.height * final_scale_h;

                // --- 坐标映射结束 ---

                if (result->num_objs >= AX_OCR_MAX_OBJ_NUM)
                    break;
                result->num_objs++;
                result->objects[result->num_objs - 1].box.center.x = rrect.center.x;
                result->objects[result->num_objs - 1].box.center.y = rrect.center.y;
                result->objects[result->num_objs - 1].box.size.w = rrect.size.width;
                result->objects[result->num_objs - 1].box.size.h = rrect.size.height;
                result->objects[result->num_objs - 1].box.angle = rrect.angle;

                result->objects[result->num_objs - 1].score = score;
                result->objects[result->num_objs - 1].orientation = orientation;
            }
        }

        return 0;
    }

private:
    float threshold = 0.3f;
};