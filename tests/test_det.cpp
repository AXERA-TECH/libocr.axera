#include "../src/models/det.hpp"
#include "cmdline.hpp"

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
    parser.add<std::string>("model", 'm', "model path", true);
    parser.add<std::string>("image", 'i', "image path", true);
    parser.parse_check(argc, argv);

    std::string model_path = parser.get<std::string>("model");
    std::string image_path = parser.get<std::string>("image");

    Det detector;
    detector.init(model_path, ax_devices.host.available ? host_device : axcl_device, 0);
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    ax_ocr_result_t result;
    int ret = detector.inference(image, &result);
    if (ret != 0)
    {
        std::cerr << "Failed to detect: " << ret << std::endl;
        return -1;
    }

    ALOGI("num_objs: %d", result.num_objs);

    for (int i = 0; i < result.num_objs; i++)
    {
        ax_ocr_obj_t &obj = result.objects[i];

        cv::RotatedRect rect = cv::RotatedRect(cv::Point2f(obj.box.center.x, obj.box.center.y), cv::Size2f(obj.box.size.w, obj.box.size.h), obj.box.angle);

        cv::Point2f vertices[4];
        rect.points(vertices);

        cv::line(image, vertices[0], vertices[1], cv::Scalar(0, 255, 0), 2);
        cv::line(image, vertices[1], vertices[2], cv::Scalar(0, 255, 0), 2);
        cv::line(image, vertices[2], vertices[3], cv::Scalar(0, 255, 0), 2);
        cv::line(image, vertices[3], vertices[0], cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("output.jpg", image);

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
