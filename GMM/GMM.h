#pragma once

class GMM {
public:
	GMM(int dimNum = 1, int mixNum = 1);
	~GMM();

	void cluster(double* data, int size);
	double getProbability(const double* x);

private:
	int m_dimNum; //����ά��
	int m_mixNum; //��˹�ֲ���Ŀ

	double* m_priors; //ÿ����˹�ֲ����������
	double** m_mean; //��ֵ(m_mixNum*m_dimNum)
	double** m_var; //����(m_mixNum*m_dimNum)

	double m_endLoss;
	int m_iterNum;

	void Estep(double* data, int size, double* w, double& cost);
	void Mstep(double* data, int size, double* w);
	double getProbability(const double* x, int j);
	void initByKMeans(double* data, int size);
};