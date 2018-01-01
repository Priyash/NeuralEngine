#ifndef CUDAEXCEPTION_H
#define CUDAEXCEPRION_H
#include<string>
#include<iostream>
#include <exception>
#include<cuda.h>
#include<cudnn.h>
#include<cublas_v2.h>
#include<string>

using namespace std;

class CudaException :public exception
{
	string msg;
public:

	CudaException(cudnnStatus_t status, string error_module);
	CudaException(cublasStatus_t status, string error_module);
	CudaException(cudaError_t status, string error_module);
	virtual const char* what()const throw();

private:
	const char* cublasGetErrorString(cublasStatus_t status);
};



















#endif