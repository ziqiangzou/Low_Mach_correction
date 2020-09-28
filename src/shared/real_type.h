/**
 * \file real_type.h
 * \brief Define macros to switch single/double precision.
 *
 * \author P. Kestener
 * \date 25-03-2010
 *
 */
#ifndef REAL_TYPE_H_
#define REAL_TYPE_H_

#include <math.h>

/**
 * \typedef real_t (alias to float or double)
 */
#ifdef USE_DOUBLE
using real_t =double;
#else
using real_t = float;
#endif // USE_DOUBLE

// math function
#if defined(USE_DOUBLE) ||  defined(USE_MIXED_PRECISION)
#define FMAX(x,y) fmax(x,y)
#define FMIN(x,y) fmin(x,y)
#define SQRT(x) sqrt(x)
#define FABS(x) fabs(x)
#define COPYSIGN(x,y) copysign(x,y)
#define ISNAN(x) isnan(x)
#define FMOD(x,y) fmod(x,y)
#define ZERO_F (0.0)
#define HALF_F (0.5)
#define ONE_FOURTH_F (0.25)
#define ONE_F  (1.0)
#define TWO_F  (2.0)
#define THREE_F  (3.0)
#define FOUR_F  (4.0)
#define ONE_THIRD_F (0.33333333333333)
#define TWO_THIRD_F (0.66666666666667)
#define ONE_THIRTIETH_F (0.03333333333333)
#define THREE_FOURTH_F (0.75)
#define ONE_TWENTIETH_F (0.05)
#define SEVEN_SIXTH_F (1.16666666666667)
#define ELEVEN_SIXTH_F (1.83333333333333)
#define ONE_SIXTH_F (0.16666666666667)
#define FIVE_SIXTH_F (0.83333333333333)
#define THIRTEEN_TWELFTH_F (1.08333333333333)
#define epsi (0.000001)
#define ZERO_ONE_F (0.1)
#define ZERO_THREE_F (0.3)
#define ZERO_SIX_F (0.6)

#else
#define FMAX(x,y) fmaxf(x,y)
#define FMIN(x,y) fminf(x,y)
#define SQRT(x) sqrtf(x)
#define FABS(x) fabsf(x)
#define COPYSIGN(x,y) copysignf(x,y)
#define ISNAN(x) isnanf(x)
#define FMOD(x,y) fmodf(x,y)
#define ZERO_F (0.0f)
#define HALF_F (0.5f)
#define ONE_FOURTH_F (0.25f)
#define ONE_F  (1.0f)
#define TWO_F  (2.0f)
#define THREE_F  (3.0f)
#define FOUR_F  (4.0f)
#define ONE_THIRD_F (0.33333333333333f)
#define TWO_THIRD_F (0.66666666666667f)
#define ONE_THIRTIETH_F (0.03333333333333f)
#define THREE_FOURTH_F (0.75f)
#define ONE_TWENTIETH_F (0.05f)
#define SEVEN_SIXTH_F (1.16666666666667f)
#define ELEVEN_SIXTH_F (1.83333333333333f)
#define ONE_SIXTH_F (0.16666666666667f)
#define FIVE_SIXTH_F (0.83333333333333f)
#define THIRTEEN_TWELFTH_F (1.08333333333333f)
#define epsi (0.000001)
#define ZERO_ONE_F (0.1f)
#define ZERO_THREE_F (0.3f)
#define ZERO_SIX_F (0.6f)
#endif // USE_DOUBLE

// other usefull macros
#define SQR(x) ((x)*(x))

#endif // REAL_TYPE_H_
