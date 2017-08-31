#pragma once

class GMM {
public:
	GMM(int dimNum = 1, int mixNum = 1);
	~GMM();

	void cluster(double* data, int size);
	double getProbability(const double* x);

private:
	int m_dimNum; //数据维数
	int m_mixNum; //高斯分布数目

	double* m_priors; //每个高斯分布的先验概率
	double** m_mean; //均值(m_mixNum*m_dimNum)
	double** m_var; //方差(m_mixNum*m_dimNum)

	double m_endLoss;
	int m_iterNum;

	void Estep(double* data, int size, double* w, double& cost);
	void Mstep(double* data, int size, double* w);
	double getProbability(const double* x, int j);
	void initByKMeans(double* data, int size);
};