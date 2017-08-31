#include <cmath>
#include <algorithm>
#include <cstdio>
#include <time.h>
#include <iostream>
#include "GMM.h"
#include "KMeans.h"
using namespace std;
#define PI 2 * asin(1.0)

GMM::GMM(int dimNum, int mixNum) {
	srand(time(NULL));

	m_dimNum = dimNum;
	m_mixNum = mixNum;

	m_priors = new double[m_mixNum];
	for (int i = 0; i < m_mixNum; i++) {
		m_priors[i] = 1.0 / m_mixNum;
	}

	m_mean = new double*[m_mixNum];
	m_var = new double*[m_mixNum];
	for (int i = 0; i < m_mixNum; i++) {
		m_mean[i] = new double[m_dimNum];
		for (int j = 0; j < m_dimNum; j++) {
			m_mean[i][j] = double(rand() % 100) / 10;
		}

		m_var[i] = new double[m_dimNum];
		for (int j = 0; j < m_dimNum; j++) {
			m_var[i][j] = double(rand() % 100) / 10;
		}
	}

	m_iterNum = 10;
	m_endLoss = 1e-3;
}

GMM::~GMM() {
	delete[] m_priors;

	for (int i = 0; i < m_mixNum; i++) {
		delete[] m_mean[i];
		delete[] m_var[i];
	}
	delete[] m_mean;
	delete[] m_var;
}

void GMM::cluster(double* data, int size) {
	initByKMeans(data, size);
	
	int iter = 0;
	double prevCost = 0.0, cost = 0.0;
	int unchanged = 0;
	bool loop = true;
	double* w = new double[size * m_mixNum];

	while (loop) {
		cost = 0.0;
		Estep(data, size, w, cost);
		Mstep(data, size, w);

		iter++;
		if (fabs(prevCost - cost) < m_endLoss * fabs(cost)) {
			unchanged++;
		}
		if (iter > m_iterNum || unchanged >= 3) {
			loop = false;
		}

		/*******for debug**********
		cout << "PRIORS: " << endl;
		for (int i = 0; i < m_mixNum; i++) {
			cout << m_priors[i] << " ";
		}
		cout << endl;
		cout << "MEANs" << endl;
		for (int i = 0; i < m_mixNum; i++) {
			for (int j = 0; j < m_dimNum; j++) {
				cout << m_mean[i][j] << " ";
			}
			cout << endl;
		}
		cout << "VARs" << endl;
		for (int i = 0; i < m_mixNum; i++) {
			for (int j = 0; j < m_dimNum; j++) {
				cout << m_var[i][j] << " ";
			}
			cout << endl;
		}
		cout << "W" << endl;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < m_mixNum; j++) {
				cout << w[i * m_mixNum + j] << " ";
			}
			cout << endl;
		}
		cout << "cost = " << cost << endl;
		********************/
	}
}

void GMM::Estep(double* data, int size, double* w, double& cost) {
	memset(w, 0, sizeof(w));

	for (int i = 0; i < size; i++) {
		double* x = new double[m_dimNum];
		for (int j = 0; j < m_dimNum; j++) {
			x[j] = data[i * m_dimNum + j];
		}

		double Ptot = getProbability(x);
		for (int j = 0; j < m_mixNum; j++) {
			w[i * m_mixNum + j] = m_priors[j] * getProbability(x, j) / Ptot;
		}
		cost += (Ptot > 1E-20) ? log10(Ptot) : -20;
	}
}

void GMM::Mstep(double* data, int size, double* w) {
	for (int i = 0; i < m_mixNum; i++) {
		double prior = 0.0;
		for (int j = 0; j < m_dimNum; j++) {
			double tot = 0.0;
			double p = 0.0;
			for (int k = 0; k < size; k++) {
				tot += w[k * m_mixNum + i] * data[k * m_dimNum + j];
				p += w[k * m_mixNum + i];
			}
			m_mean[i][j] = tot / p;
			prior += p;
		}
		m_priors[i] = prior / m_dimNum / size;
	}

	for (int i = 0; i < m_mixNum; i++) {
		for (int j = 0; j < m_dimNum; j++) {
			double tot = 0.0;
			double p = 0.0;
			for (int k = 0; k < size; k++) {
				tot += w[k * m_mixNum + i] * pow((data[k * m_dimNum + j] - m_mean[i][j]), 2.0);
				p += w[k * m_mixNum + i];
			}
			//m_var[i][j] = tot / p;
			m_var[i][j] = max(tot / p, 0.2);
			//测试数据比较特殊，会出现方差为0的情况，使计算机计算错误，所以设个阈值。实际中
			//由于数据量较大，这一项不一定需要。
		}
	}
}

void GMM::initByKMeans(double* data, int size) {
	KMeans* kmeans = new KMeans(m_dimNum, m_mixNum);
	int* Labels = new int[size];
	kmeans->cluster(data, size, Labels);

	int* counts = new int[m_mixNum];
	memset(counts, 0, sizeof(int) * m_mixNum);
	double** sum = new double*[m_mixNum];
	for (int i = 0; i < m_mixNum; i++) {
		sum[i] = new double[m_dimNum];
		memset(sum[i], 0, sizeof(double)*m_dimNum);
	}
	double** ssum = new double*[m_mixNum];
	for (int i = 0; i < m_mixNum; i++) {
		ssum[i] = new double[m_dimNum];
		memset(ssum[i], 0, sizeof(double)*m_dimNum);
	}

	for (int i = 0; i < size; i++) {
		counts[Labels[i]]++;
		for (int j = 0; j < m_dimNum; j++) {
			sum[Labels[i]][j] += data[i * m_dimNum + j];
			ssum[Labels[i]][j] += pow(data[i * m_dimNum + j], 2.0);
		}
	}

	for (int i = 0; i < m_mixNum; i++) {
		m_priors[i] = double(counts[i]) / size;
		for (int j = 0; j < m_dimNum; j++) {
			m_mean[i][j] = sum[i][j] / counts[i];
			m_var[i][j] = ssum[i][j] / counts[i] - pow(m_mean[i][j], 2.0);
			m_var[i][j] = max(m_var[i][j], 0.2);
			//测试数据比较特殊，会出现方差为0的情况，使计算机计算错误，所以设个阈值。实际中
			//由于数据量较大，这一项不一定需要。
		}
	}

	delete kmeans;
	delete[] Labels;
	delete[] counts;
	for (int i = 0; i < m_mixNum; i++) {
		delete[] sum[i];
		delete[] ssum[i];
	}
	delete[] sum;
	delete[] ssum;
}


double GMM::getProbability(const double* x) {
	double p = 0.0;
	for (int i = 0; i < m_mixNum; i++) {
		p += m_priors[i] * getProbability(x, i);
	}
	return p;
}

double GMM::getProbability(const double* x, int j) {
	double p = 1.0;
	for (int i = 0; i < m_dimNum; i++) {
		p *= 1.0 / sqrt(2 * PI * m_var[j][i]);
		p *= exp(-0.5 * ((x[i] - m_mean[j][i]) * (x[i] - m_mean[j][i])) / m_var[j][i]);
	}
	return p;
}
