#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
using namespace cv;

const OrtApi* Ort::g_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

int main(int argc, char* argv[])
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
/*
	std::cout<< argv[0]<<std::endl;
	std::cout<< argv[1]<<std::endl;
	std::cout<< argv[2]<<std::endl;
	std::cout<< argv[3]<<std::endl;
	std::cout<< argc <<std::endl;*/

	char model_path[256], input_path[1000], output_path[1000];
	if(argc < 2)
		strcpy(input_path, "../NASA_MSLMHL_0009_EXCERPT/test/images_s/0046MH0000090010100121I01_DRCX_s.png");
	else
		strcpy(input_path, argv[1]);
	if(argc < 3)
		strcpy(output_path, "../output_cpp.png");
	else
		strcpy(output_path, argv[2]);
	if(argc < 4)
		strcpy(model_path, "../nasa_srmodel.onnx");
	else
		strcpy(model_path, argv[3]);

	// initialize session options if needed
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

	Ort::Session session(env, model_path, session_options);

 	std::vector<const char*> output_node_names = {"output"};
	std::vector<const char*> input_node_names = {"input"};

	Mat image, out_img;
	std::vector<cv::Mat> yCrCbChannels(3);
	image = imread(input_path);
	cvtColor(image, image, CV_BGR2YCrCb);
	split(image, yCrCbChannels);

	size_t input_tensor_size = 1 * 1 * 72 * 96;
	std::vector<float> input_tensor_values(input_tensor_size);

	for(int i = 0; i < 72; i++)
		for(int j = 0; j < 96; j++)
			input_tensor_values[i*96+j] = (float)(yCrCbChannels[0].at<uchar>(i, j))/255.0;

  // Get input node shape
  std::vector<int64_t> input_node_dims;
	Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	input_node_dims = tensor_info.GetShape();

	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

 	// get back output tensor
	auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());


	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	assert(abs(floatarr[0] - 0.000045) < 1e-6);

  resize(yCrCbChannels[0], yCrCbChannels[0], Size(96*2, 72*2));
  resize(yCrCbChannels[1], yCrCbChannels[1], Size(96*2, 72*2), INTER_CUBIC);
  resize(yCrCbChannels[2], yCrCbChannels[2], Size(96*2, 72*2), INTER_CUBIC);

	int tmpindex;
	for(int i = 0; i < 144; i++)
	{
		for(int j = 0; j < 192; j++)
		{
			tmpindex = i*192+j;
			floatarr[tmpindex] = floatarr[tmpindex]*255.0;
			if(floatarr[tmpindex] > 255)
				floatarr[tmpindex] = 255;
			if(floatarr[tmpindex] < 0)
				floatarr[tmpindex] = 0;
			yCrCbChannels[0].at<uchar>(i, j) = (uchar)floatarr[tmpindex];
		}
	}

	std::vector<cv::Mat> merge_img;
	merge_img.push_back(yCrCbChannels[0]);
	merge_img.push_back(yCrCbChannels[1]);
	merge_img.push_back(yCrCbChannels[2]);
	merge(merge_img, out_img);
	cvtColor(out_img, out_img, CV_YCrCb2BGR);
	imwrite(output_path, out_img);
	return 0;
}
