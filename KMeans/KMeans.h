#pragma once

class KMeans {
public:
	KMeans(int dimNum = 1, int clueterNum = 1);
	~KMeans();

	void cluster(double* data, int size, int* Labels);
private:
	int m_dimNum; //数据维度
	int m_clusterNum; //聚类数目
	double** m_means; //聚类中心

	int m_iterNum; //迭代次数
	double m_endLoss; //终止条件

	double GetLabel(const double* x, int& label);
	double GetDistance(const double* u, const double* v, int dimNum);

	void Estep(double* data, int size, int* Labels, double& cost);
	void Mstep(double* data, int size, int* Labels);
};