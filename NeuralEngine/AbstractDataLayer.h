#ifndef ABSTRACTDATALAYER_H
#define ABSTRACTDATALAYER_H

class AbstractDataLayer
{
public:
	AbstractDataLayer();
	~AbstractDataLayer();

	virtual void allocateCpuData(float*) = 0;
	virtual float* fillData() = 0;
	virtual void allocateGpuData(float*) = 0;
	virtual float* copyDataToGpu() = 0;
	virtual float* copyDataToCpu() = 0;
};





#endif