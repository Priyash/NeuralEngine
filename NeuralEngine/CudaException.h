#ifndef CUDAEXCEPTION_H
#define CUDAEXCEPRION_H
#include<string>
#include<iostream>
#include <exception>
#include<cudnn.h>
#include<string>

using namespace std;

class CudaException :public exception
{
	string msg;
public:

	CudaException(cudnnStatus_t status, string error_module);
	virtual const char* what()const throw();
};



















#endif