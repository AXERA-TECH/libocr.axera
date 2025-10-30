#ifndef __LIBOCR_H__
#define __LIBOCR_H__
#include "ax_devices.h"

#if defined(__cplusplus)
extern "C"
{
#endif
#define AX_OCR_MAX_OBJ_NUM 64
#define AX_OCR_MAX_TEXT_LEN 256
    typedef enum
    {
        ax_ocr_errcode_failed = -1,
        ax_ocr_errcode_success = 0,
    } ax_ocr_errcode_e;

    typedef struct
    {
        int width;
        int height;
        int channels;
        int stride;
        void *data;
    } ax_ocr_img_t;

    typedef struct
    {
        struct
        {
            struct
            {
                int x, y;
            } center;
            struct
            {
                int w, h;
            } size;

            float angle;
        } box;

        float score;
        int orientation;
        struct
        {
            int token;
            float score;
        } tokens[AX_OCR_MAX_TEXT_LEN];
        int num_tokens;
        char text[AX_OCR_MAX_TEXT_LEN];
    } ax_ocr_obj_t;

    typedef struct
    {
        ax_ocr_obj_t objects[AX_OCR_MAX_OBJ_NUM];
        int num_objs;
    } ax_ocr_result_t;

    typedef struct
    {
        ax_devive_e dev_type; // Device type
        char devid;           // axcl device ID

        char det_model_path[256];
        char cls_model_path[256];
        char rec_model_path[256];

        char rec_charset_path[256];
    } ax_ocr_init_t;

    typedef void *ax_ocr_handle_t;

    int ax_ocr_init(ax_ocr_init_t *init, ax_ocr_handle_t *handle);
    int ax_ocr_deinit(ax_ocr_handle_t handle);

    int ax_ocr(ax_ocr_handle_t handle, ax_ocr_img_t *img, ax_ocr_result_t *result);

#if defined(__cplusplus)
}
#endif
#endif
