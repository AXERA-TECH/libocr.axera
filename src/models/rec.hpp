#pragma once
#include "cls.hpp"
#include <opencv2/opencv.hpp>

class Rec : public Cls
{
public:
    int inference(cv::Mat input, ax_ocr_result_t *result) override
    {
        cv::Mat padded = resize_img(input, input_height, input_width);

        // char filename[256];
        // snprintf(filename, sizeof(filename), "rec_img_%d.jpg", cur_idx);
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

        int len = m_runner->get_output(0).vShape[1];
        int total_tokens = m_runner->get_output(0).vShape[2];

        // 40 x 18385

        int last_token = 0;

        result->objects[cur_idx].num_tokens = 0;

        for (int i = 0; i < len; i++)
        {
            const float *p = output_data + i * total_tokens;

            int index = 0;
            float max_score = -9999.f;
            for (int j = 0; j < total_tokens; j++)
            {
                float score = p[j];
                if (score > max_score)
                {
                    max_score = score;
                    index = j;
                }
            }

            if (last_token == index) // CTC rule, if index is same as last one, they will be merged into one token
                continue;

            last_token = index;

            if (index <= 0)
                continue;

            if (result->objects[cur_idx].num_tokens >= AX_OCR_MAX_TEXT_LEN)
                break;

            result->objects[cur_idx].tokens[result->objects[cur_idx].num_tokens].token = index - 1;
            result->objects[cur_idx].tokens[result->objects[cur_idx].num_tokens].score = max_score;
            result->objects[cur_idx].num_tokens++;
        }

        return 0;
    }
};