// Copyright (c) 2001,2004  INRIA Sophia-Antipolis (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL$
// $Id$
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Sylvain Pion

#ifndef CGAL_INTERNAL_STATIC_FILTERS_ORIENTATION_3_H
#define CGAL_INTERNAL_STATIC_FILTERS_ORIENTATION_3_H
#define CGAL_USE_SSE2_MAX

#include <CGAL/Profile_counter.h>
#include <CGAL/Filtered_kernel/internal/Static_filters/Static_filter_error.h>
#include <cmath>

namespace CGAL { namespace internal { namespace Static_filters_predicates {


#include <iostream>


template < typename K_base >
class Orientation_3
  : public K_base::Orientation_3
{
  typedef typename K_base::Point_3          Point_3;
  typedef typename K_base::Vector_3         Vector_3;
  typedef typename K_base::Sphere_3         Sphere_3;
  typedef typename K_base::Tetrahedron_3    Tetrahedron_3;
  typedef typename K_base::Orientation_3    Base;

public:
 typedef typename Base::result_type  result_type;

  using Base::operator();

  result_type
  operator()(const Point_3 &p, const Point_3 &q,
             const Point_3 &r, const Point_3 &s) const
  {
      CGAL_BRANCH_PROFILER_3("semi-static failures/attempts/calls to   : Orientation_3", tmp);

      double px, py, pz, qx, qy, qz, rx, ry, rz, sx, sy, sz;

      if (fit_in_double(p.x(), px) && fit_in_double(p.y(), py) &&
          fit_in_double(p.z(), pz) &&
          fit_in_double(q.x(), qx) && fit_in_double(q.y(), qy) &&
          fit_in_double(q.z(), qz) &&
          fit_in_double(r.x(), rx) && fit_in_double(r.y(), ry) &&
          fit_in_double(r.z(), rz) &&
          fit_in_double(s.x(), sx) && fit_in_double(s.y(), sy) &&
          fit_in_double(s.z(), sz))
      {
          CGAL_BRANCH_PROFILER_BRANCH_1(tmp);

          double pqx = qx - px;
          double pqy = qy - py;
          double pqz = qz - pz;
          double prx = rx - px;
          double pry = ry - py;
          double prz = rz - pz;
          double psx = sx - px;
          double psy = sy - py;
          double psz = sz - pz;

          // CGAL::abs uses fabs on platforms where it is faster than (a<0)?-a:a
          // Then semi-static filter.

          double maxx = CGAL::abs(pqx);
          double maxy = CGAL::abs(pqy);
          double maxz = CGAL::abs(pqz);
#ifdef CGAL_USE_SSE2_MAX
          {
            __m128d tx = _mm_max_sd(_mm_set_sd(maxx), _mm_set_sd(abs(prx)));
            maxx = _mm_cvtsd_f64(_mm_max_sd(tx, _mm_set_sd(abs(psx))));

            __m128d ty = _mm_max_sd(_mm_set_sd(maxy), _mm_set_sd(abs(pry)));
            maxy = _mm_cvtsd_f64(_mm_max_sd(ty, _mm_set_sd(abs(psy))));

            __m128d tz = _mm_max_sd(_mm_set_sd(maxz), _mm_set_sd(abs(prz)));
            maxz = _mm_cvtsd_f64(_mm_max_sd(tz, _mm_set_sd(abs(psz))));
          }
#else
          if (maxx < aprx) maxx = aprx;
          if (maxx < apsx) maxx = apsx;
          if (maxy < apry) maxy = apry;
          if (maxy < apsy) maxy = apsy;
          if (maxz < aprz) maxz = aprz;
          if (maxz < apsz) maxz = apsz;
#endif
          double det = CGAL::determinant(pqx, pqy, pqz,
                                         prx, pry, prz,
                                         psx, psy, psz);

          double eps = 5.1107127829973299e-15 * maxx * maxy * maxz;

#ifdef CGAL_USE_SSE2_MAX
          {
            __m128d vx = _mm_set_sd(maxx);
            __m128d vy = _mm_set_sd(maxy);
            __m128d vz = _mm_set_sd(maxz);

            // pairwise reduction
            __m128d min_xy = _mm_min_sd(vx, vy);
            __m128d max_xy = _mm_max_sd(vx, vy);

            __m128d min_all = _mm_min_sd(min_xy, vz);
            __m128d max_all = _mm_max_sd(max_xy, vz);

            maxx = _mm_cvtsd_f64(min_all);
            maxz = _mm_cvtsd_f64(max_all);
            // maxy intentionally ignored
          }
#else
          // Sort maxx < maxy < maxz.
          if (maxx > maxz)
              std::swap(maxx, maxz);
          if (maxy > maxz)
              std::swap(maxy, maxz);
          else if (maxy < maxx)
              std::swap(maxx, maxy);
#endif
          // Protect against underflow in the computation of eps.
          if (maxx < 1e-97) /* cbrt(min_double/eps) */ {
            if (maxx == 0)
              return ZERO;
          }
          // Protect against overflow in the computation of det.
          else if (maxz < 1e102) /* cbrt(max_double [hadamard]/4) */ {

            if (det > eps)  return POSITIVE;
            if (det < -eps) return NEGATIVE;
          }
          CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
      }

      return Base::operator()(p, q, r, s);
  }

  // Computes the epsilon for Orientation_3.
  static double compute_epsilon()
  {
    typedef Static_filter_error F;
    F t1 = F(1, F::ulp()/2);         // First translation
    F det = CGAL::determinant(t1, t1, t1,
                              t1, t1, t1,
                              t1, t1, t1); // Full det
    double err = det.error();
    err += err * 2 * F::ulp(); // Correction due to "eps * maxx * maxy...".
    std::cerr << "*** epsilon for Orientation_3 = " << err << std::endl;
    return err;
  }

};

} } } // namespace CGAL::internal::Static_filters_predicates

#endif // CGAL_INTERNAL_STATIC_FILTERS_ORIENTATION_3_H
