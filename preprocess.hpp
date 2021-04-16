#pragma once
#include "common.hpp"


std::vector<uint8_t> Preprocess(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size, const ScaleType scale)
{
    // Image channels are in BGR order. Currently model configuration
    // data doesn't provide any information as to the expected channel
    // orderings (like RGB, BGR). We are going to assume that RGB is the
    // most likely ordering and so change the channels to that ordering.
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
    cv::Mat sample_resized;
    if (sample.size() != img_size)
    {
        cv::resize(sample, sample_resized, img_size);
    }
    else
    {
        sample_resized = sample;
    }

    cv::Mat sample_type;
    sample_resized.convertTo(
        sample_type, (img_channels == 3) ? img_type3 : img_type1);
   
    cv::Mat sample_final;
    sample.convertTo(
        sample_type, (img_channels == 3) ? img_type3 : img_type1);
    const int INPUT_W = 608;
    const int INPUT_H = 608;
    int w, h, x, y;
    float r_w = INPUT_W / (sample_type.cols * 1.0);
    float r_h = INPUT_H / (sample_type.rows * 1.0);
    if (r_h > r_w)
    {
        w = INPUT_W;
        h = r_w * sample_type.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    }
    else
    {
        w = r_h * sample_type.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(sample_type, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    out.convertTo(sample_final, CV_32FC3, 1.f / 255.f);


    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample_final.total() * sample_final.elemSize();
    size_t pos = 0;
    input_data.resize(img_byte_size);

    // (format.compare("FORMAT_NCHW") == 0)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'input_data'. The BGR channels are
    // backed by the 'input_data' vector so that ends up with CHW
    // order of the data.
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i)
    {
        input_bgr_channels.emplace_back(
            img_size.height, img_size.width, img_type1, &(input_data[pos]));
        pos += input_bgr_channels.back().total() *
            input_bgr_channels.back().elemSize();
    }

    cv::split(sample_final, input_bgr_channels);

    if (pos != img_byte_size)
    {
        std::cerr << "unexpected total size of channels " << pos << ", expecting "
            << img_byte_size << std::endl;
        exit(1);
    }

    return input_data;
}