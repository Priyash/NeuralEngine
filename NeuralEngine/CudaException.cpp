#include"CudaException.h"


CudaException::CudaException(cudnnStatus_t status, string error_module)
{
	
	string s(cudnnGetErrorString(status));
	msg = "Error on line " + __LINE__ + s + " for " + error_module;
}

CudaException::CudaException(cublasStatus_t status, string error_module)
{

	string s(cublasGetErrorString(status));
	msg = "Error on line " + __LINE__ + s + " for " + error_module;
	int k = 0;
}

CudaException::CudaException(cudaError_t status, string error_module)
{

	string s(cudaGetErrorString(status));
	msg = "Error on line " + __LINE__ + s + " for " + error_module;
	int k = 0;
}

const char* CudaException::what()const throw()
{
	return msg.c_str();
}



const char* CudaException::cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "UNKNOWN_CUBLAS_ERROR";
}