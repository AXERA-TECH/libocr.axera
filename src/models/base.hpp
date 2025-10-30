#pragma once
#include <memory>
#include "../runner/ax650/ax_model_runner_ax650.hpp"
#include "../runner/axcl/ax_model_runner_axcl.hpp"
#include "../utils/mmap.hpp"
#include "libocr.h"

#include <opencv2/opencv.hpp>

class base_model
{
protected:
    std::shared_ptr<ax_runner_base> m_runner;
    int input_width = 0;
    int input_height = 0;

    int cur_idx = -1;

public:
    base_model() = default;
    virtual ~base_model() = default;

    virtual int init(std::string model_path, ax_devive_e dev_type, int devid)
    {
        MMap image_mmap(model_path.c_str());

        if (dev_type == ax_devive_e::host_device)
        {
            m_runner = std::make_shared<ax_runner_ax650>();
            auto ret = m_runner->init(image_mmap.data(), image_mmap.size(), -1);
            if (ret != 0)
            {
                ALOGE("text encoder init failed\n");
                return -1;
            }
        }
        else if (dev_type == ax_devive_e::axcl_device)
        {
            m_runner = std::make_shared<ax_runner_axcl>();
            auto ret = m_runner->init(image_mmap.data(), image_mmap.size(), devid);
            if (ret != 0)
            {
                ALOGE("text encoder init failed\n");
                return -1;
            }
        }
        else
        {
            ALOGE("dev_type not support\n");
            return -1;
        }

        if (m_runner->get_input(0).vShape[1] == 3)
        {
            input_height = m_runner->get_input(0).vShape[2];
            input_width = m_runner->get_input(0).vShape[3];
        }
        else
        {
            input_height = m_runner->get_input(0).vShape[1];
            input_width = m_runner->get_input(0).vShape[2];
        }
        ALOGI("input_width: %d, input_height: %d\n", input_width, input_height);

        return 0;
    }

    virtual int deinit()
    {
        m_runner->deinit();
        return 0;
    }

    virtual int set_affinity(int id)
    {
        return m_runner->set_affinity(id);
    }

    void set_cur_idx(int idx)
    {
        cur_idx = idx;
    }

    virtual int inference(cv::Mat input, ax_ocr_result_t *result) = 0;
};