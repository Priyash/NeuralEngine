#include<iostream>
#include"ImageManager.h"
#include<cudnn.h>
#include <cassert>
#include <cstdlib>
#include"CudaException.h"
#include <cuda.h>



void save_image(const char* output_filename,
	float* buffer,
	int height,
	int width) {
	cv::Mat output_image(height, width, CV_32FC3, buffer);
	// Make negative values zero.
	cv::threshold(output_image,
		output_image,
		/*threshold=*/0,
		/*maxval=*/0,
		cv::THRESH_TOZERO);
	cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
	output_image.convertTo(output_image, CV_8UC3);
	string out_file;
	Util::getInstance()->read_config_file("config.txt");
	Util::getInstance()->parse_config_file(CONFIG_ID::IMAGE_OUTPUT_PATH);
	out_file = Util::getInstance()->getImageOutputPath();
	string out(output_filename);
	out_file.append(out);
	cv::imwrite(out_file, output_image);
	std::cerr << "Write output to " << output_filename << std::endl;
}


void check_cuda_status(cudnnStatus_t status , string error_module)
{
	if (status != CUDNN_STATUS_SUCCESS)
	{
		throw CudaException(status, error_module);
	}
	return;
}

int main()
{

	ImageManager* manager = new ImageManager();
	vector<Mat>data = manager->getImageMatrices(IMAGE::NORMAL); 
	cout << data.size() << endl;
	Mat image = data[0];
	cudaSetDevice(1);
	cudnnStatus_t status;

	

	try
	{
		cudnnHandle_t cudnn;
		status = cudnnCreate(&cudnn);
		check_cuda_status(status, "cudnnHandleCreateError");


		cudnnTensorDescriptor_t input_descriptor;
		status = cudnnCreateTensorDescriptor(&input_descriptor);
		check_cuda_status(status , "input_descriptor");
		status = cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, image.rows, image.cols);
		check_cuda_status(status , "set_input_descriptor");
		

		cudnnFilterDescriptor_t kernel_descriptor;
		status = cudnnCreateFilterDescriptor(&kernel_descriptor);
		check_cuda_status(status , "kernel_descriptor");
		status = cudnnSetFilter4dDescriptor(kernel_descriptor,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*out_channels=*/3,
			/*in_channels=*/3,
			/*kernel_height=*/3,
			/*kernel_width=*/3);
		check_cuda_status(status , "set_kernel_descriptor");

		cudnnConvolutionDescriptor_t convolution_descriptor;
		status = cudnnCreateConvolutionDescriptor(&convolution_descriptor);
		check_cuda_status(status , "convolution_descriptor");
		status = cudnnSetConvolution2dDescriptor(convolution_descriptor,
			/*pad_height=*/1,
			/*pad_width=*/1,
			/*vertical_stride=*/1,
			/*horizontal_stride=*/1,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/CUDNN_CROSS_CORRELATION,
			/*computeType=*/CUDNN_DATA_FLOAT);
		check_cuda_status(status,"set_convolution_descriptor");

		int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
		status = cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
			input_descriptor,
			kernel_descriptor,
			&batch_size,
			&channels,
			&height,
			&width);

		check_cuda_status(status , "cudnnGetConvolution2dForwardOutputDim");
		std::cerr << "Output Image: " << height << " x " << width << " x " << channels
			<< std::endl;

		cudnnTensorDescriptor_t output_descriptor;
		status = cudnnCreateTensorDescriptor(&output_descriptor);
		check_cuda_status(status,"output_descriptor");
		cudnnSetTensor4dDescriptor(output_descriptor,
			/*format=*/CUDNN_TENSOR_NHWC,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/3,
			/*image_height=*/image.rows,
			/*image_width=*/image.cols);
		
		check_cuda_status(status,"set_output_descriptor");

		cudnnConvolutionFwdAlgo_t convolution_algorithm;
		status = cudnnGetConvolutionForwardAlgorithm(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			0,
			&convolution_algorithm);
		check_cuda_status(status , "cudnnGetConvolutionForwardAlgorithm");

		size_t workspace_bytes = 0;
		status = cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			convolution_algorithm,
			&workspace_bytes);
		check_cuda_status(status , "cudnnGetConvolutionForwardWorkspaceSize");
		std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
			<< std::endl;
		assert(workspace_bytes > 0);

		void* d_workspace{ nullptr };
		cudaMalloc(&d_workspace, workspace_bytes);

		int image_bytes = batch_size * channels * height * width * sizeof(float);

		float* d_input{ nullptr };
		cudaMalloc(&d_input, image_bytes);
		cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

		float* d_output{ nullptr };
		cudaMalloc(&d_output, image_bytes);
		cudaMemset(d_output, 0, image_bytes);


		// clang-format off
		const float kernel_template[3][3] = {
			{ 1, 1, 1 },
			{ 1, -8, 1 },
			{ 1, 1, 1 }
		};


		float h_kernel[3][3][3][3];
		for (int kernel = 0; kernel < 3; ++kernel) {
			for (int channel = 0; channel < 3; ++channel) {
				for (int row = 0; row < 3; ++row) {
					for (int column = 0; column < 3; ++column) {
						h_kernel[kernel][channel][row][column] = kernel_template[row][column];
					}
				}
			}
		}

		float* d_kernel{ nullptr };
		cudaMalloc(&d_kernel, sizeof(h_kernel));
		cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

		const float alpha = 1.0f, beta = 0.0f;

		status = cudnnConvolutionForward(cudnn,
			&alpha,
			input_descriptor,
			d_input,
			kernel_descriptor,
			d_kernel,
			convolution_descriptor,
			convolution_algorithm,
			d_workspace,
			workspace_bytes,
			&beta,
			output_descriptor,
			d_output);
		check_cuda_status(status,"cudnnConvolutionForward");


		cudnnActivationDescriptor_t activation_descriptor;
		status = cudnnCreateActivationDescriptor(&activation_descriptor);
		check_cuda_status(status,"cudnnCreateActivationDescriptor");
		status = cudnnSetActivationDescriptor(activation_descriptor,
			CUDNN_ACTIVATION_SIGMOID,
			CUDNN_PROPAGATE_NAN,
			/*relu_coef=*/0);
		check_cuda_status(status , "cudnnSetActivationDescriptor");
		status = cudnnActivationForward(cudnn,
			activation_descriptor,
			&alpha,
			output_descriptor,
			d_output,
			&beta,
			output_descriptor,
			d_output);
		check_cuda_status(status,"cudnnActivationForward");
		cudnnDestroyActivationDescriptor(activation_descriptor);
		float* h_output = new float[image_bytes];
		cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

		save_image("cudnn-out.png", h_output, height, width);

		delete[] h_output;
		cudaFree(d_kernel);
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_workspace);

		cudnnDestroyTensorDescriptor(input_descriptor);
		cudnnDestroyTensorDescriptor(output_descriptor);
		cudnnDestroyFilterDescriptor(kernel_descriptor);
		cudnnDestroyConvolutionDescriptor(convolution_descriptor);

		cudnnDestroy(cudnn);


	}
	catch (CudaException& ce)
	{
		cout << ce.what() << endl;
	}


	return 0;
}