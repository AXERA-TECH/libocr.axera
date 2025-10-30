#include "libocr.h"
#include "cmdline.hpp"
#include "timer.hpp"
#include <fstream>
#include <cstring>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
    }

    if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_init(axcl_device, 0);
    }

    if (!ax_devices.host.available && ax_devices.devices.count == 0)
    {
        printf("no device available\n");
        return -1;
    }

    cmdline::parser parser;
    parser.add<std::string>("det", 0, "det model path", true);
    parser.add<std::string>("cls", 0, "cls model path", true);
    parser.add<std::string>("rec", 0, "rec model path", true);
    parser.add<std::string>("dict", 0, "character dict path", true);

    parser.add<std::string>("image", 'i', "image folder(jpg png etc....)", true);
    parser.add<std::string>("output", 'o', "output", false, "results.jpg");
    parser.parse_check(argc, argv);

    ax_ocr_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    if (ax_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else if (ax_devices.devices.count > 0)
    {
        init_info.dev_type = axcl_device;
        init_info.devid = 0;
    }

    sprintf(init_info.det_model_path, "%s", parser.get<std::string>("det").c_str());
    sprintf(init_info.cls_model_path, "%s", parser.get<std::string>("cls").c_str());
    sprintf(init_info.rec_model_path, "%s", parser.get<std::string>("rec").c_str());
    sprintf(init_info.rec_charset_path, "%s", parser.get<std::string>("dict").c_str());

    ax_ocr_handle_t handle;
    int ret = ax_ocr_init(&init_info, &handle);
    if (ret != ax_ocr_errcode_success)
    {
        printf("ax_ocr_init failed\n");
        return -1;
    }

    std::string image_src = parser.get<std::string>("image");
    cv::Mat src = cv::imread(image_src);
    if (src.empty())
    {
        printf("imread %s failed\n", image_src.c_str());
        return -1;
    }
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    ax_ocr_img_t img;
    img.data = src.data;
    img.width = src.cols;
    img.height = src.rows;
    img.channels = src.channels();
    img.stride = src.step;
    ax_ocr_result_t result;
    memset(&result, 0, sizeof(result));
    ret = ax_ocr(handle, &img, &result);
    if (ret != ax_ocr_errcode_success)
    {
        printf("ax_ocr failed\n");
        return -1;
    }
    printf("num_objs: %d\n", result.num_objs);

    cv::cvtColor(src, src, cv::COLOR_RGB2BGR);

    for (int i = 0; i < result.num_objs; i++)
    {
        ax_ocr_obj_t &obj = result.objects[i];
        cv::RotatedRect rrect = cv::RotatedRect(cv::Point2f(obj.box.center.x, obj.box.center.y), cv::Size2f(obj.box.size.w, obj.box.size.h), obj.box.angle);
        cv::Point2f vertices[4];
        rrect.points(vertices);
        for (int j = 0; j < 4; j++)
        {
            cv::line(src, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        printf("obj %03d: num_tokens %03d, text %s\n", i, obj.num_tokens, obj.text);
    }

    cv::imwrite(parser.get<std::string>("output"), src);

    ax_ocr_deinit(handle);

    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_deinit(axcl_device, 0);
    }

    return 0;
}