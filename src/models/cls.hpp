#pragma once
#include "base.hpp"
#include <opencv2/opencv.hpp>

class Cls : public base_model
{
protected:
    cv::Mat resize_img(cv::Mat &img, int imgH, int imgW)
    {
        int h = img.rows;
        int w = img.cols;
        float ratio = w / float(h);
        int resized_w = 0;
        if (std::ceil(imgH * ratio) > imgW)
        {
            resized_w = imgW;
        }
        else
        {
            resized_w = int(std::ceil(imgH * ratio));
        }
        cv::Mat resized_img(imgH, resized_w, img.type());
        cv::resize(img, resized_img, cv::Size(resized_w, imgH));
        cv::Mat padded_img(imgH, imgW, img.type(), cv::Scalar(0));
        cv::Rect roi(0, 0, resized_w, imgH);
        resized_img(roi).copyTo(padded_img(roi));
        return padded_img;
    }

public:
    int inference(cv::Mat input, ax_ocr_result_t *result) override
    {
        cv::Mat padded = resize_img(input, input_height, input_width);
        
        // char filename[256];
        // snprintf(filename, sizeof(filename), "cls_img_%d.jpg", cur_idx);
        // cv::imwrite(filename, padded);

        float *input_data = (float *)m_runner->get_input(0).pVirAddr;
        unsigned char *img_data = (unsigned char *)padded.data;

        int letterbox_cols = padded.cols;
        int letterbox_rows = padded.rows;

        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    input_data[out_index] = float(img_data[in_index]);
                }
            }
        }
        m_runner->inference();
        float *output_data = (float *)m_runner->get_output(0).pVirAddr;
        if (output_data[1] > output_data[0] && output_data[1] > 0.9)
        {
            return 1;
        }

        return 0;
    }
};