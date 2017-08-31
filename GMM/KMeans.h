#pragma once

class KMeans {
public:
	KMeans(int dimNum = 1, int clueterNum = 1);
	~KMeans();

	void cluster(double* data, int size, int* Labels);
private:
	int m_dimNum; //����ά��
	int m_clusterNum; //������Ŀ
	double** m_means; //��������

	int m_iterNum; //��������
	double m_endLoss; //��ֹ����

	double GetLabel(const double* x, int& label);
	double GetDistance(const double* u, const double* v, int dimNum);

	void Estep(double* data, int size, int* Labels, double& cost);
	void Mstep(double* data, int size, int* Labels);
};