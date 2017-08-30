#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <time.h>
#include <iostream>
#include "KMeans.h"
using namespace std;

KMeans::KMeans(int dimNum, int clusterNum) {
	srand((unsigned)time(NULL));

	m_dimNum = dimNum;
	m_clusterNum = clusterNum;

	m_means = new double*[m_clusterNum];
	for (int i = 0; i < m_clusterNum; i++) {
		m_means[i] = new double[m_dimNum];
		for (int j = 0; j < m_dimNum; j++) {
			m_means[i][j] = double(rand() % 10000) / 1000;
		}
	}

	m_iterNum = 100;
	m_endLoss = 1e-3;
}

KMeans::~KMeans() {
	for (int i = 0; i < m_clusterNum; i++) {
		delete[] m_means[i];
	}
	delete[] m_means;
}

void KMeans::cluster(double* data, int size, int* Labels) {
	//数据展开为一位向量输入
	assert(size >= m_clusterNum);

	double prevCost = 0.0;
	double cost = 0.0;
	int unchanged = 0;
	int iter = 0;
	bool loop = true;
	while (loop) {
		Estep(data, size, Labels, cost);
		Mstep(data, size, Labels);
		
		/**
		if (iter % 1 == 0) {
			cout << "run " << iter << " rounds" << endl;
			cout << "prevCost = " << prevCost << " cost = " << cost << endl;
			for (int i = 0; i < size; i++) {
				cout << "point " << i << " belongs to " << Labels[i] << endl;
			}
			for (int i = 0; i < m_clusterNum; i++) {
				for (int j = 0; j < m_dimNum; j++) {
					cout << m_means[i][j] << " ";
				}
				cout << endl;
			}
		}
		*/
		
		if (fabs(cost - prevCost) < m_endLoss * prevCost) {
			unchanged++;
		}
		if (iter > m_iterNum || unchanged >= 3) {
			loop = false;
		}
		prevCost = cost;
		cost = 0.0;
		iter++;
	}

}

void KMeans::Estep(double* data, int size, int* Labels, double& cost) {
	double* x = new double[m_dimNum];

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < m_dimNum; j++) {
			x[j] = data[i * m_dimNum + j];
		}

		int label;
		cost += GetLabel(x, label);
		Labels[i] = label;
	}
	cost /= size;
	delete[] x;
}

void KMeans::Mstep(double* data, int size, int* Labels) {
	double** sum = new double*[m_clusterNum];
	for (int i = 0; i < m_clusterNum; i++) {
		sum[i] = new double[m_dimNum];
		memset(sum[i], 0, sizeof(double) * m_dimNum);
	}
	int* num = new int[m_clusterNum];
	memset(num, 0, sizeof(int) * m_clusterNum);

	for (int i = 0; i < size; i++) {
		num[Labels[i]]++;
		for (int j = 0; j < m_dimNum; j++) {
			sum[Labels[i]][j] += data[i * m_dimNum + j];
		}
	}

	for (int i = 0; i < m_clusterNum; i++) {
		if (num[i] > 0) {
			for (int j = 0; j < m_dimNum; j++) {
				m_means[i][j] = sum[i][j] / num[i];
			}
		}
	}

	for (int i = 0; i < m_clusterNum; i++) {
		delete[] sum[i];
	}
	delete[] sum;
	delete[] num;
}

double KMeans::GetLabel(const double* x, int& label) {
	label = 0;
	double minDis = GetDistance(x, m_means[0], m_dimNum);
	for (int i = 1; i < m_clusterNum; i++) {
		double dis = GetDistance(x, m_means[i], m_dimNum);
		if (dis < minDis) {
			minDis = dis; label = i;
		}
	}
	return minDis;
}

double KMeans::GetDistance(const double* u, const double* v, int dimNum) {
	double dist = 0.0;
	for (int i = 0; i < dimNum; i++) {
		dist += (u[i] - v[i]) * (u[i] - v[i]);
	}
	return sqrt(dist);
}