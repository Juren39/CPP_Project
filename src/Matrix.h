#pragma once
//
// Created by 胡鸿飞 on 2021/5/30.
//
//
// Created by 胡鸿飞 on 2021/5/30.
//

#ifndef CPROJECT_MATRIX_H
#define CPROJECT_MATRIX_H



#include <vector>
#include <complex>
#include <typeinfo>
#include <cmath>
#include <map>
#include <ostream>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
template<typename T>
class Vector;
template<typename T>
class Matrix {
    friend class Vector<T>;
private:
    vector<vector<T>> vec;
public:


    explicit Matrix() : Matrix(0, 0) {};
    explicit Matrix(int rows, int cols);


    //Copy Constructor
    Matrix(const Matrix<T>& matrix);

    // 移动构造函数
    Matrix(Matrix<T>&& matrix) noexcept;

    //Copy Assignment operator
    Matrix<T>& operator=(const Matrix<T>& matrix);

    // Move Assignment operator
    Matrix<T>& operator=(Matrix<T>&& matrix) noexcept;

    explicit Matrix<T>(const vector<vector<T>>& vec);

    explicit Matrix<T>(vector<vector<T>>&& vec);

    static Matrix<T> values(int rows = 0, int cols = 0, T = static_cast<T>(0));

    static Matrix<T> eye(int s);

    static Matrix<T> eye_value(int s, T t);



    void xiugai(int i, int j, int x);
    int chakan(int rows, int cols);


    friend std::ostream& operator<<(std::ostream& output, const Matrix<T>& matrix) {
        for (const auto& i : matrix.vec) {
            for (const auto& j : i) {
                output << j << " ";
            }
            output << endl;
        }
        return output;
    }

    Matrix operator+(Matrix<T> &right) {
        Matrix<T> m1(vec.size(), vec[0].size());
        if (vec.size() != right.vec.size() && vec[0].size() != right.vec[0].size()) {
            std::cout << "The dimension is wrong!" << endl;
            return m1;
        }

        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, vec[i][j] + right.vec[i][j]);
            }
        }
        return m1;
    }//add

    Matrix operator-(Matrix<T> &right) {
        Matrix<T> m1(vec.size(), vec[0].size());
        if (vec.size() != right.vec.size() && vec[0].size() != right.vec[0].size()) {
            std::cout << "The dimension is wrong!" << endl;
            return m1;
        }
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, vec[i][j] - right.vec[i][j]);
            }
        }
        return m1;
    }//sub

    Matrix operator*(int num) {
        Matrix<T> m1(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, this->getvecvalue(i, j) * num);
            }
        }
        return m1;
    }//scalar multiplication

    Matrix operator/(int num) {
        Matrix<T> m1(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, this->getvecvalue(i, j) / num);
            }
        }
        return m1;
    }//scalar division

    Matrix Transpose() {
        Matrix<T> m1(vec[0].size(), vec.size());
        for (int i = 0; i < vec[0].size(); i++) {
            for (int j = 0; j < vec.size(); j++) {
                m1.setVecValue(i, j, this->getvecvalue(j, i));
            }
        }
        return m1;
    }//transpose

    Matrix Conjugation() {
        Matrix<T> m1(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                complex<T> m2;
                m2.real(real(this->getvecvalue(i, j)));
                m2.imag(imag(this->getvecvalue(i, j)) * -1);
                m1.setVecValue(i, j, m2);
            }
        }
        return m1;
    }//Conjugation

    Matrix Element_Wise(const Matrix<T> &right) {
        if (vec.size() != right.vec.size() || vec[0].size() != right.vec[0].size()) {
            cout << "the dimension is wrong!" << endl;
        }
        Matrix<T> m1(vec.size(), vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[0].size(); j++) {
                m1.setVecValue(i, j, vec[i][j] * right.vec[i][j]);
            }
        }
        return m1;
    }//Element_Wise

    Matrix operator*(Matrix<T> &right) {
        if (vec[0].size() != right.vec.size()) {
            std::cout << "The dimension is wrong!" << endl;
            return this;
        }
        Matrix<T> m1(this->vec.size(), right.vec[0].size());
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < right.vec[0].size(); j++) {
                for (int k = 0; k < vec[0].size(); k++) {
                    m1.setVecValue(i, j, this->getvecvalue(i, k) * right.getvecvalue(k, j));
                }
            }
        }
        return m1;
    }//matrix-matrix multiplication

    Matrix operator*(Vector<T> &right) {
        if (this->vec[0].size() != right.dimension) {
            cout << "The dimension is wrong!" << endl;
            return 0;
        }
        Matrix<T> m1(right.dimension, 1);
        for (int i = 0; i < right.dimension; i++) {
            m1 = m1 + this->pickColvalue(this, i) * right[i];
        }
        return m1;
    }//matrix-vector multiplication

    static inline T hypot(T a, T b)
    {
        a = std::abs(a);
        b = std::abs(b);
        if (a > b) {
            b /= a;
            return a*std::sqrt(1 + b*b);
        }
        if (b > 0) {
            a /= b;
            return b*std::sqrt(1 + a*a);
        }
        return 0;
    }

    int eigen(vector<T>& eigenvalues, vector<vector<T>>& eigenvectors, bool sort_ = true)
    {
        auto n = vec.size();
        for (const auto& m : vec) {
            if (m.size() != n) {
                fprintf(stderr, "mat must be square and it should be a real symmetric matrix\n");
                return -1;
            }
        }

        eigenvalues.resize(n, (T)0);
        std::vector<T> V(n * n, (T)0);
        for (int i = 0; i < n; ++i) {
            V[n * i + i] = (T)1;
            eigenvalues[i] = vec[i][i];
        }

        const T eps = std::numeric_limits<T>::epsilon();
        int maxIters{ (int)n * (int)n * 30 };
        T mv{(T)0 };
        std::vector<int> indR(n, 0), indC(n, 0);
        std::vector<T> A;
        for (int i = 0; i < n; ++i) {
            A.insert(A.begin() + i * n, vec[i].begin(), vec[i].end());
        }

        for (int k = 0; k < n; ++k) {
            int m, i;
            if (k < n - 1) {
                for (m = k + 1, mv = std::abs(A[n*k + m]), i = k + 2; i < n; i++) {
                    T val = std::abs(A[n * k + i]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indR[k] = m;
            }
            if (k > 0) {
                for (m = 0, mv = std::abs(A[k]), i = 1; i < k; i++) {
                    T val = std::abs(A[n * i + k]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indC[k] = m;
            }
        }

        if (n > 1) for (int iters = 0; iters < maxIters; iters++) {
                int k, i, m;
                // find index (k,l) of pivot p
                for (k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n - 1; i++) {
                    T val = std::abs(A[n * i + indR[i]]);
                    if (mv < val)
                        mv = val, k = i;
                }
                int l = indR[k];
                for (i = 1; i < n; i++) {
                    T val = std::abs(A[n * indC[i] + i]);
                    if (mv < val)
                        mv = val, k = indC[i], l = i;
                }

                T p = A[n * k + l];
                if (std::abs(p) <= eps)
                    break;
                T y = (T)((eigenvalues[l] - eigenvalues[k]) * 0.5);
                T t = std::abs(y) + hypot(p, y);
                T s = hypot(p, t);
                T c = t / s;
                s = p / s; t = (p / t)*p;
                if (y < 0)
                    s = -s, t = -t;
                A[n*k + l] = 0;

                eigenvalues[k] -= t;
                eigenvalues[l] += t;

                T a0, b0;

#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c

                // rotate rows and columns k and l
                for (i = 0; i < k; i++)
                    rotate(A[n*i + k], A[n*i + l]);
                for (i = k + 1; i < l; i++)
                    rotate(A[n*k + i], A[n*i + l]);
                for (i = l + 1; i < n; i++)
                    rotate(A[n*k + i], A[n*l + i]);

                // rotate eigenvectors
                for (i = 0; i < n; i++)
                    rotate(V[n*k+i], V[n*l+i]);

#undef rotate

                for (int j = 0; j < 2; j++) {
                    int idx = j == 0 ? k : l;
                    if (idx < n - 1) {
                        for (m = idx + 1, mv = std::abs(A[n*idx + m]), i = idx + 2; i < n; i++) {
                            T val = std::abs(A[n * idx + i]);
                            if (mv < val)
                                mv = val, m = i;
                        }
                        indR[idx] = m;
                    }
                    if (idx > 0) {
                        for (m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++) {
                            T val = std::abs(A[n * i + idx]);
                            if (mv < val)
                                mv = val, m = i;
                        }
                        indC[idx] = m;
                    }
                }
            }

        // sort eigenvalues & eigenvectors
        if (sort_) {
            for (int k = 0; k < n - 1; k++) {
                int m = k;
                for (int i = k + 1; i < n; i++) {
                    if (eigenvalues[m] < eigenvalues[i])
                        m = i;
                }
                if (k != m) {
                    std::swap(eigenvalues[m], eigenvalues[k]);
                    for (int i = 0; i < n; i++)
                        std::swap(V[n*m+i], V[n*k+i]);
                }
            }
        }

        eigenvectors.resize(n);
        for (int i = 0; i < n; ++i) {
            eigenvectors[i].resize(n);
            eigenvectors[i].assign(V.begin() + i * n, V.begin() + i * n + n);
        }

        return 0;
    }//eigenvalues and eigenvectors

    T Traces(){
        T traces;
        for (int i = 0; i < vec.size(); i++) {
            traces += vec[i][i];
        }
        return traces;
    }//traces

    T determinant(vector<vector<T>> det , int size)//det-行列式，n:行列式的阶数
    {
        T detVal = 0;//行列式的值

        if(size == 1)//递归终止条件
            return det[0][0];

        vector<vector<T>> tempdet = vector<vector<T >>(size - 1 , vector<T>( size - 1 ));


        for(int i=0;i < size ;i++)//第一重循环，行列式按第一行展开
        {
            for(int j = 0 ; j < size -1 ; j++)
                for(int k=0;k < size - 1 ;k++)
                {
                    if(k <i)
                        tempdet[j][k] = det[j+1][k] ;
                    else
                        tempdet[j][k] = det[j+1][k+1];
                }
            detVal += det[0][i] * pow(-1 , i ) * determinant(tempdet, size - 1 );

        }
        return detVal;
    }//determinant




    static Matrix<T> reshape(Matrix<T> m, int r, int c);
    static Matrix<T> slicing(Matrix<T> m, int rb, int re, int cb, int ce);
    static Matrix<T> convolution(Matrix<T> m1, Matrix<T> m2);


    void zhuanhuan(Matrix<T> m);
    static Matrix<T> nzh();


    inline int rows() const;

    inline int cols() const;
};

template<typename T>
class Vector {
private:
    vector<vector<T>> value;
    int dimension{}; //向量的维数
public:
    Vector()= default;;
    Vector(Vector& source);
    void setValue(int i,T newvalue);
    T getValue(int i);
    void setDimension(int newDimension);
    int getDimension();

    Vector pickvalue(Matrix<T> & m, int col){
        Vector<T> m1;
        m1.dimension = m.vec[0].size();
        for(int i = 0 ; i < m1.dimension ; i++){
            m1[i] = m.getvecvalue(col , i);
        }
        return m1;
    }
    Vector operator +(Vector& right)
    {
        if(dimension != right.dimension)
        {
            std::cout << "The dimension is wrong!" << endl;
            return this;
        }
        for(int i = 0 ; i < dimension ; i++){
            this->setValue(i,this->getValue(i) + right.getValue(i));
        }
        return this;
    }
    Vector operator -(Vector& right)
    {
        if(dimension != right.dimension)
        {
            std::cout << "The dimension is wrong!" << endl;
            return this;
        }
        for(int i = 0 ; i < dimension ; i++){
            this->setValue(i,this->getValue(i) - right.getValue(i));
        }
        return this;
    }

    T operator *(Vector& right)
    {
        T ans;
        for(int i = 0 ; i < dimension ; i++){
            ans += this->getValue(i) * right.getValue(i);
        }
        return ans;
    }//dot product

    Matrix<T> operator *(Matrix<T>& right)
    {
        if(dimension != right.vec.size()){
            cout << "The dimension is wrong!" << endl;
            return 0;
        }
        Matrix<T> m1(1 , right.dimension);
        for(int i = 0 ; i < right.dimension ; i++){
            m1 = m1 + this->pickRowvalue(this , i) * right[i];
        }
        return m1;
    }//matrix-vector multiplication

    friend std::ostream &operator<<(std::ostream &output, const Vector<T>& v) {
        for (const auto &i : v) {
            output << i << " ";
        }
        output << endl;
        return output;
    }
};

template<typename T>
void Vector<T>::setValue(int i, T newvalue) {
    value[i] = newvalue;
}

template<typename T>
Vector<T>::Vector(Vector &source) {
    dimension = source.dimension;
    for(int i = 0 ; i < dimension ; i++){
        value[i] = source.value[i];
    }
}

template<typename T>
T Vector<T>::getValue(int i) {
    return value[i];
}

template<typename T>
void Vector<T>::setDimension(int newDimension) {
    dimension = newDimension;
}

template<typename T>
int Vector<T>::getDimension() {
    return dimension;
}

template<typename T>
int Matrix<T>::chakan(int rows, int cols) {
    return this->vec[rows][cols];
}


template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
    this->vec = vector<vector<T >>(rows, vector<T>(cols));
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& matrix) {
    this->vec = vector<vector<T >>(matrix.vec);
}

template<typename T>
void Matrix<T>::xiugai(int i, int j, int x) {
    this->vec[i][j] = x;
}



template<typename T>
Matrix<T>::Matrix(Matrix<T>&& matrix) noexcept : vec(matrix.vec) {
    matrix.vec = vector<vector<T>>{ 0, vector<T>{0, static_cast<T>(0)} };
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& matrix) {
    this->vec = matrix.vec;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& matrix) noexcept {
    this->vec = std::move(matrix.vec);
    matrix.vec = vector<vector<T>>{ 0, vector<T>{0, static_cast<T>(0)} };
    return *this;
}

template<typename T>
Matrix<T>::Matrix(const vector<vector<T>>& vec) {
    this->vec = vec;

}

template<typename T>
Matrix<T>::Matrix(vector<vector<T>>&& vec) {
    this->vec = std::move(vec);
}

template<typename T>
Matrix<T> Matrix<T>::values(int rows, int cols, T t) {
    Matrix<T> will_return(rows, cols);
    for (auto& i : will_return.vec) {
        i = vector<T>(cols, t);
    }
    return will_return;
}

template<typename T>
inline int Matrix<T>::rows() const {
    return static_cast<int>(this->vec.size());
}

template<typename T>
inline int Matrix<T>::cols() const {
    if (this->rows() == 0) {
        return 0;
    }
    return static_cast<int>(this->vec.front().size());
}


template<typename T>
Matrix<T> reshape(Matrix<T> m, int row, int col) {
    int row1 = m.rows();
    int col1 = m.cols();
    if (row1 * col1 < row * col) {
        return m;
    }
    else {
        vector<T> array;
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col1; j++) {
                array.push_back(m.chakan(i, j));
            }
        }
        int now = 0;
        Matrix<T> s(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                s.xiugai(i, j, array[now]);
                now++;


            }
        }
        now = 0;
        return s;
    }
}

template<typename T>
Matrix<T> slicing(Matrix<T> m, int rowbegin, int rowend, int colbegin, int colend) {
    int row1 = rowend - rowbegin + 1;
    int col1 = colend - colbegin + 1;
    Matrix<T> s(row1, col1);
    if (rowbegin > rowend || colbegin > colend) {
        cout << "Wrong!" << endl;
    }
    else {
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col1; j++) {
                s.xiugai(i, j, m.chakan(i + rowbegin - 1, j + colbegin - 1));

            }
        }
    }
    return s;
}



template<typename T>
Matrix<T> Matrix<T>::convolution(Matrix<T> m1, Matrix<T> m2) {
    Matrix<T> big;
    Matrix<T> small;
    if (m1.rows() * m1.cols() > m2.rows() * m2.cols()) {
        big = m1;
        small = m2;
    }
    else {
        big = m2;
        small = m1;
    }

    Matrix<T> extend(big.rows() + 2 * (small.rows() - 1), big.cols() + 2 * (small.cols() - 1));
    Matrix<T> ans(big.rows() + small.rows() - 1, big.cols() + small.cols() - 1);
    Matrix<T> daosmall(small.rows(), small.cols());
    for (int i = 0; i < small.rows(); ++i) {
        for (int j = 0; j < small.cols(); ++j) {
            daosmall.xiugai(small.rows() - i - 1, small.cols() - j - 1, small.chakan(i, j));
        }
    }//将小矩阵逆转
    for (int i = 0; i < big.rows(); ++i) {
        for (int j = 0; j < big.cols(); ++j) {
            extend.xiugai(small.rows() - 1 + i, small.cols() - 1 + j, big.chakan(i, j));

        }
    }//将大矩阵扩大至完全能够涵盖小矩阵的运算空间

    for (int i = 0; i < ans.rows(); ++i) {
        for (int j = 0; j < ans.cols(); ++j) {
            T sum = 0;
            for (int k = 0; k < daosmall.rows(); ++k) {
                for (int l = 0; l < daosmall.cols(); ++l) {
                    sum += daosmall.chakan(k, l) * extend.chakan(i + k, j + l);
                }
            }
            ans.xiugai(i, j, sum);

        }
    } //daosmall矩阵起点在extend矩阵的位置，对应ans矩阵的i,j
    return ans;
}


template<typename T>
void zhuanhuan(Matrix<T> m) {
    int rows = m.rows();
    int cols = m.cols();
    Mat testMat1 = Mat(Size(cols, rows), CV_32FC1);
    string name;
    cout << "请输入储存文件名：";
    cin >> name;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            testMat1.at<float>(i, j) = m.chakan(i,j);
        }
    }
    string name1 = "./" + name + ".xml";

    FileStorage fs(name1, FileStorage::WRITE);
    fs << "mat1" << testMat1;
    fs.release();
}

template<typename T>
Matrix<T>  Matrix<T>::nzh() {
    cout << "请输入储存文件名称： ";
    string name;
    cin >> name;
    string name1 = "./" + name + ".xml";

    FileStorage fsRead(name1, FileStorage::READ);
    Mat readMat1;
    fsRead["mat1"] >> readMat1;

    Matrix<T> m(readMat1.rows, readMat1.cols);
    for (int i = 0; i < readMat1.rows; i++)
    {
        for (int j = 0; j < readMat1.cols; j++)
        {
            m.xiugai(i, j, readMat1.at<float>(i, j));
        }


    }

    return m;
}



#endif //MATRIX_H
