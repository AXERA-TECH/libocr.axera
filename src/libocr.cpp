#include "libocr.h"

#include "models/det.hpp"
#include "models/rec.hpp"
#include "models/cls.hpp"

#include <memory>
#include <fstream>

struct ax_ocr_handle_internal_t
{
    Det det;
    Rec rec;
    Cls cls;

    std::vector<std::string> character_dict;
};

static cv::Mat get_rotate_crop_image(const cv::Mat &rgb, const ax_ocr_obj_t &object)
{
    const int orientation = object.orientation;
    const float rw = object.box.size.w;
    const float rh = object.box.size.h;

    const int target_height = 48;
    const float target_width = rh * target_height / rw;

    // warpperspective shall be used to rotate the image
    // but actually they are all rectangles, so warpaffine is almost enough  :P

    cv::Mat dst;

    cv::Point2f corners[4];
    // object.rrect.points(corners);
    cv::RotatedRect rrect = cv::RotatedRect(cv::Point2f(object.box.center.x, object.box.center.y), cv::Size2f(rw, rh), object.box.angle);
    rrect.points(corners);

    if (orientation == 0)
    {
        // horizontal text
        // corner points order
        //  0--------1
        //  |        |rw  -> as angle=90
        //  3--------2
        //      rh

        std::vector<cv::Point2f> src_pts(3);
        src_pts[0] = corners[0];
        src_pts[1] = corners[1];
        src_pts[2] = corners[3];

        std::vector<cv::Point2f> dst_pts(3);
        dst_pts[0] = cv::Point2f(0, 0);
        dst_pts[1] = cv::Point2f(target_width, 0);
        dst_pts[2] = cv::Point2f(0, target_height);

        cv::Mat tm = cv::getAffineTransform(src_pts, dst_pts);

        cv::warpAffine(rgb, dst, tm, cv::Size(target_width, target_height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }
    else
    {
        // vertial text
        // corner points order
        //  1----2
        //  |    |
        //  |    |
        //  |    |rh  -> as angle=0
        //  |    |
        //  |    |
        //  0----3
        //    rw

        std::vector<cv::Point2f> src_pts(3);
        src_pts[0] = corners[2];
        src_pts[1] = corners[3];
        src_pts[2] = corners[1];

        std::vector<cv::Point2f> dst_pts(3);
        dst_pts[0] = cv::Point2f(0, 0);
        dst_pts[1] = cv::Point2f(target_width, 0);
        dst_pts[2] = cv::Point2f(0, target_height);

        cv::Mat tm = cv::getAffineTransform(src_pts, dst_pts);

        cv::warpAffine(rgb, dst, tm, cv::Size(target_width, target_height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }

    return dst;
}

int ax_ocr_init(ax_ocr_init_t *init, ax_ocr_handle_t *handle)
{
    ax_ocr_handle_internal_t *internal = new ax_ocr_handle_internal_t();

    internal->det.init(init->det_model_path, init->dev_type, init->devid);
    internal->cls.init(init->cls_model_path, init->dev_type, init->devid);
    internal->rec.init(init->rec_model_path, init->dev_type, init->devid);

    std::ifstream ifs(init->rec_charset_path);
    if (!ifs.is_open())
    {
        ALOGE("open rec_charset_path failed: %s", init->rec_charset_path);
        return -1;
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        internal->character_dict.push_back(line);
    }

    *handle = internal;
    return 0;
}

int ax_ocr_deinit(ax_ocr_handle_t handle)
{
    ax_ocr_handle_internal_t *internal = (ax_ocr_handle_internal_t *)handle;

    internal->det.deinit();
    internal->rec.deinit();
    internal->cls.deinit();

    delete internal;
    return 0;
}

int ax_ocr(ax_ocr_handle_t handle, ax_ocr_img_t *img, ax_ocr_result_t *result)
{
    ax_ocr_handle_internal_t *internal = (ax_ocr_handle_internal_t *)handle;

    cv::Mat img_cv = cv::Mat(img->height, img->width, CV_8UC(img->channels), img->data, img->stride);

    cv::Mat input;
    switch (img->channels)
    {
    case 4:
        cv::cvtColor(img_cv, input, cv::COLOR_BGRA2BGR);
        break;
    case 1:
        cv::cvtColor(img_cv, input, cv::COLOR_GRAY2BGR);
        break;
    case 3:
        input = img_cv;
        break;
    default:
        ALOGE("only support channel 1,3,4 uint8 image");
        return -1;
    }

    internal->det.inference(input, result);

    for (size_t i = 0; i < result->num_objs; i++)
    {
        ax_ocr_obj_t &obj = result->objects[i];
        cv::Mat crop_img = get_rotate_crop_image(input, obj);

        // char filename[256];
        // snprintf(filename, sizeof(filename), "crop_img_%d.jpg", i);
        // cv::imwrite(filename, crop_img);

        internal->cls.set_cur_idx(i);
        int label = internal->cls.inference(crop_img, result);
        if (label == 1)
        {
            cv::rotate(crop_img, crop_img, cv::ROTATE_180);
        }
        internal->rec.set_cur_idx(i);
        internal->rec.inference(crop_img, result);

        std::string text;
        for (size_t j = 0; j < obj.num_tokens; j++)
        {
            if (obj.tokens[j].token >= internal->character_dict.size())
            {
                if (!text.empty() && text.back() != ' ')
                {
                    text += " ";
                }
                continue;
            }

            text += internal->character_dict[obj.tokens[j].token];
            // if (obj.orientation == 0)
            // {
            // }
            // else
            // {
            //     text += internal->character_dict[obj.tokens[j].token];
            //     if (j + 1 < obj.num_tokens)
            //         text += "\n";
            // }
        }
        strcpy(result->objects[i].text, text.c_str());
    }

    return 0;
}
