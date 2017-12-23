#include"CudaException.h"


CudaException::CudaException(cudnnStatus_t status, string error_module)
{
	
	string s(cudnnGetErrorString(status));
	msg = "Error on line " + __LINE__ + s + " for " + error_module;
	int k = 0;
}

const char* CudaException::what()const throw()
{
	return msg.c_str();
}