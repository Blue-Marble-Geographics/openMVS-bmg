/****************************************************************************
* VCGLib                                                            o o     *
* Visual and Computer Graphics Library                            o     o   *
*                                                                _   O  _   *
* Copyright(C) 2004-2016                                           \/)\/    *
* Visual Computing Lab                                            /\/|      *
* ISTI - Italian National Research Council                           |      *
*                                                                    \      *
* All rights reserved.                                                      *
*                                                                           *
* This program is free software; you can redistribute it and/or modify      *
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
****************************************************************************/
#ifndef __VCGLIB_QUADRIC
#define __VCGLIB_QUADRIC

#include <vcg/space/point3.h>
#include <vcg/space/plane3.h>
#include <vcg/math/matrix33.h>
#include <Eigen/Core>

// This is not breaking clean.
#define MINIMUM_IMPROVEMENT // JPB WIP

namespace vcg {
namespace math {

/*
 *  This class encode a quadric function 
 *  f(x) = xAx +bx + c
 *  where A is a symmetric 3x3 matrix, b a vector and c a scalar constant.  
 */ 
template<typename  _ScalarType>
class Quadric
{
public:
  typedef _ScalarType ScalarType;
  union {
    struct {
      ScalarType a[6];		// Symmetric Matrix 3x3 : a11 a12 a13 a22 a23 a33
      ScalarType b[3];		// Vector r3
      ScalarType c;			  // Scalar (-1 means null/un-initialized quadric)
    };
    ScalarType array[10];
  };

  inline Quadric() { c = -1; }

  void Print() const
  {
    std::cout << "A: ";
    for (int i = 0; i < 6; ++i) {
      std::cout << a[i] << " ";
    }

    std::cout << "B: ";
    for (int i = 0; i < 3; ++i) {
      std::cout << b[i] << " ";
    }

    std::cout << "C: " << c << "\n";
  }
  
  bool IsValid() const { return c>=0; }
  void SetInvalid() { c = -1.0; }
  
   // Initialize the quadric to keep the squared distance from a given Plane  
  template< class PlaneType >
  void ByPlane( const PlaneType & p )
  {
    a[0] =  (ScalarType)p.Direction()[0]*p.Direction()[0];	// a11
    a[1] =  (ScalarType)p.Direction()[1]*p.Direction()[0];	// a12 (=a21)
    a[2] =  (ScalarType)p.Direction()[2]*p.Direction()[0];	// a13 (=a31)
    a[3] =  (ScalarType)p.Direction()[1]*p.Direction()[1];	// a22
    a[4] =  (ScalarType)p.Direction()[2]*p.Direction()[1];	// a23 (=a32)
    a[5] =  (ScalarType)p.Direction()[2]*p.Direction()[2];	// a33
    b[0] =  (ScalarType)(-2.0)*p.Offset()*p.Direction()[0];
    b[1] =  (ScalarType)(-2.0)*p.Offset()*p.Direction()[1];
    b[2] =  (ScalarType)(-2.0)*p.Offset()*p.Direction()[2];
    c    =  (ScalarType)p.Offset()*p.Offset();
  }
  
  /* 
   * Initializes the quadric as the squared distance from a given line.
   * Note that this code also works for a vcg::Ray<T>, even though the (squared) distance
   * from a ray is different "before" its origin.
   */
  template< class LineType >
  void ByLine( const LineType & r ) // Init dato un raggio
  {
    ScalarType K = (ScalarType)(r.Origin()*r.Direction());
    a[0] = (ScalarType)1.0-r.Direction()[0]*r.Direction()[0]; // a11
    a[1] = (ScalarType)-r.Direction()[0]*r.Direction()[1]; // a12 (=a21)
    a[2] = (ScalarType)-r.Direction()[0]*r.Direction()[2]; // a13 (=a31)
    a[3] = (ScalarType)1.0-r.Direction()[1]*r.Direction()[1]; // a22
    a[4] = (ScalarType)-r.Direction()[1]*r.Direction()[2]; // a23 (=a32)
    a[5] = (ScalarType)1.0-r.Direction()[2]*r.Direction()[2]; // a33
    b[0] = (ScalarType)2.0*(r.Direction()[0]*K - r.Origin()[0]);
    b[1] = (ScalarType)2.0*(r.Direction()[1]*K - r.Origin()[1]);
    b[2] = (ScalarType)2.0*(r.Direction()[2]*K - r.Origin()[2]);
    c = -K*K + (ScalarType)(r.Origin()*r.Origin());
  }

  /*
   * Initializes the quadric as the squared distance from a given point.
   *
   */
  template< class CoordType >
  void ByPoint( const CoordType & p ) // Init dato un raggio
  {
    a[0] = 1; // a11
    a[1] = 0; // a12 (=a21)
    a[2] = 0; // a13 (=a31)
    a[3] = 1; // a22
    a[4] = 0; // a23 (=a32)
    a[5] = 1; // a33
    b[0] = (ScalarType)-2.0*((ScalarType)p.X());
    b[1] = (ScalarType)-2.0*((ScalarType)p.Y());
    b[2] = (ScalarType)-2.0*((ScalarType)p.Z());
    c = pow(p.X(),2) + pow(p.Y(),2)+ pow(p.Z(),2);
  }

  void SetZero()
  {
    a[0] = 0;
    a[1] = 0;
    a[2] = 0;
    a[3] = 0;
    a[4] = 0;
    a[5] = 0;
    b[0] = 0;
    b[1] = 0;
    b[2] = 0;
    c    = 0;
  }
  
  void operator = ( const Quadric & q )
  {
    assert( q.IsValid() );
    
    a[0] = q.a[0];
    a[1] = q.a[1];
    a[2] = q.a[2];
    a[3] = q.a[3];
    a[4] = q.a[4];
    a[5] = q.a[5];
    b[0] = q.b[0];
    b[1] = q.b[1];
    b[2] = q.b[2];
    c    = q.c;
  }

  void operator += ( const Quadric & q )
  {
    assert( IsValid() );
    assert( q.IsValid() );
    
    a[0] += q.a[0];
    a[1] += q.a[1];
    a[2] += q.a[2];
    a[3] += q.a[3];
    a[4] += q.a[4];
    a[5] += q.a[5];
    b[0] += q.b[0];
    b[1] += q.b[1];
    b[2] += q.b[2];
    c    += q.c;
  }
 
  void operator *= ( const ScalarType & w )			// Amplifica una quadirca
  {
    assert( IsValid() );
    
    a[0] *= w;
    a[1] *= w;
    a[2] *= w;
    a[3] *= w;
    a[4] *= w;
    a[5] *= w;
    b[0] *= w;
    b[1] *= w;
    b[2] *= w;
    c    *= w;
  }
  
  /* Evaluate a quadric over a point p.
   */
  template <class ResultScalarType>
  ResultScalarType Apply( const Point3<ResultScalarType> & p ) const
  {
    assert( IsValid() );
    return ResultScalarType (
            p[0]*p[0]*a[0] + 2*p[0]*p[1]*a[1] + 2*p[0]*p[2]*a[2] + p[0]*b[0]
        +   p[1]*p[1]*a[3] + 2*p[1]*p[2]*a[4] + p[1]*b[1]
        +   p[2]*p[2]*a[5] + p[2]*b[2]	+ c);
  }
  
  static double &RelativeErrorThr()
  {
    static double _err = 0.000001;
    return _err;
  }
  
  // Find the point minimizing the quadric xAx + bx + c 
  // by solving the first derivative 2 Ax + b = 0 
  // return true if the found solution fits the system. 
  
  template <class ReturnScalarType>
  bool Minimum(Point3<ReturnScalarType> &x) const
  {
#ifdef MINIMUM_IMPROVEMENT
#if 1
    // Matrix A (symmetric)
    bool success = false;

    const ReturnScalarType A00 = a[0], A01 = a[1], A02 = a[2];
    const ReturnScalarType      A11 = a[3], A12 = a[4];
    const ReturnScalarType           A22 = a[5];

    // Vector b (divided by 2)
    const ReturnScalarType b0 = -b[0] * ReturnScalarType(0.5);
    const ReturnScalarType b1 = -b[1] * ReturnScalarType(0.5);
    const ReturnScalarType b2 = -b[2] * ReturnScalarType(0.5);

    // Cholesky decomposition (A = L * Lt)
    ReturnScalarType L00 = FastSqrtS(A00);
    auto invL00 = 1.0 / L00;

    ReturnScalarType L10 = A01 * invL00;
    ReturnScalarType L20 = A02 * invL00;

    ReturnScalarType d11 = A11 - L10 * L10;

    const ReturnScalarType L11 = FastSqrtS(d11);
    const auto invL11 = ReturnScalarType(1.0) / L11;

    ReturnScalarType L21 = (A12 - L10 * L20) * invL11;

    ReturnScalarType d22 = A22 - L20 * L20 - L21 * L21;
    ReturnScalarType L22 = FastSqrtS(d22);
    auto invL22 = ReturnScalarType(1.0) / L22;

    // Forward substitution: solve L * y = b
    ReturnScalarType y0 = b0 * invL00;
    ReturnScalarType y1 = (b1 - L10 * y0) * invL11;
    ReturnScalarType y2 = (b2 - L20 * y0 - L21 * y1) * invL22;

    // Backward substitution: solve Lt * x = y
    ReturnScalarType x2 = y2 * invL22;
    ReturnScalarType x1 = (y1 - L21 * x2) * invL11;
    ReturnScalarType x0 = (y0 - L10 * x1 - L20 * x2) * invL00;

    x[0] = x0;
    x[1] = x1;
    x[2] = x2;

    return (L00 >= 1e-12) && (d11 > 0.) && (d22 > 0.);
#else
    using Scalar = ReturnScalarType;
    using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

    Mat3 A;
    A << a[0], a[1], a[2],
         a[1], a[3], a[4],
         a[2], a[4], a[5];

    Vec3  be(-b[0] / Scalar(2), -b[1] / Scalar(2), -b[2] / Scalar(2));

    Eigen::LLT<Mat3> solver(A);

    if (solver.info() != Eigen::Success) {
      return false;
      ++failures;
    }

    Vec3 xe = solver.solve(be);

    x.FromEigenVector(xe);
    return true;
#endif
#else
    Eigen::Matrix3d A;
    Eigen::Vector3d be;
    A << a[0], a[1], a[2],
         a[1], a[3], a[4],
         a[2], a[4], a[5];
    be << -b[0]/2, -b[1]/2, -b[2]/2;
  
  //  Eigen::Vector3d xe = A.colPivHouseholderQr().solve(bv);
  //  Eigen::Vector3d xe = A.partialPivLu().solve(bv);
    Eigen::Vector3d xe = A.fullPivLu().solve(be);
    double relative_error = (A*xe - be).norm() / be.norm();
    if(relative_error> Quadric<ScalarType>::RelativeErrorThr() ) 
      return false;
    
    x.FromEigenVector(xe);
    return true;
#endif
  }
  
  
  template <class ReturnScalarType>
  bool MinimumClosestToPoint(Point3<ReturnScalarType> &x, const Point3<ReturnScalarType> &pt)
  {
    const double qeps = 1e-3;
    Eigen::Matrix3d A;
    Eigen::Vector3d be;
    A << a[0], a[1], a[2],
         a[1], a[3], a[4],
         a[2], a[4], a[5];
    be << -b[0]/2, -b[1]/2, -b[2]/2;
  
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector3d s = svd.singularValues();
    for(int i=1;i<3;++i) 
      if(s[i]/s[0] > qeps) s[i]=1/s[i];
      else s[i]=0;
    s[0]=1/s[0];
    
    Eigen::Vector3d xp;  
    pt.ToEigenVector(xp);
    Eigen::Vector3d xe = xp + (svd.matrixV()*s.asDiagonal()*(svd.matrixU().transpose())) *(be - A*xp);

    x.FromEigenVector(xe);
    return true;
  }
  
};

template<typename T>
__forceinline Quadric<T> operator + (const Quadric<T>& lhs, const Quadric<T>& rhs)
{
  auto result = lhs;   // make a copy of lhs
  result += rhs;          // reuse operator+=
  return result;
}

typedef Quadric<short>  Quadrics;
typedef Quadric<int>	  Quadrici;
typedef Quadric<float>  Quadricf;
typedef Quadric<double> Quadricd;



	} // end namespace math
} // end namespace vcg

#endif
