#pragma once

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor3D.h"
#include "shared/RiemannSolvers.h"

// init conditions
#include "shared/BlastParams.h"

namespace euler_kokkos { namespace all_regime
	{

		class ComputeAcousticStepFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeAcousticStepFunctor3D(HydroParams params_,
						DataArray Udata_, DataArrayConst Qdata_, DataArray gradphi_,
						real_t dt_) :
					HydroBaseFunctor3D(params_), Udata(Udata_), Qdata(Qdata_),gradphi(gradphi_), m_K(params.settings.K),
					dtdx(dt_/params.dx), dtdy(dt_/params.dy), dtdz(dt_/params.dz),
					half_dtdx(HALF_F * dtdx), half_dtdy(HALF_F * dtdy), half_dtdz(HALF_F * dtdz),
					conservative(params.settings.conservative) {};

				static void apply(HydroParams params,
						DataArray Udata, DataArrayConst Qdata, DataArray gradphi,
						real_t dt, int nbCells)
				{
					ComputeAcousticStepFunctor3D functor(params, Udata, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void computeAcousticRelaxation(const HydroState& qLoc, real_t cLoc,
							const HydroState& qNei, real_t cNei,
							real_t M, int IX, int dir,
							real_t& uStar, real_t& piStar) const
					{
						const real_t a = m_K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
						uStar = dir * HALF_F * (qNei[IX] + qLoc[IX]) - HALF_F * (qNei[IP] - qLoc[IP] + M) / a;

						const real_t theta = params.settings.low_mach_correction ? FMIN(abs(uStar) / FMAX(cNei, cLoc), ONE_F) : ONE_F;
						piStar = + HALF_F * (qNei[IP] + qLoc[IP]) - dir * HALF_F * theta * a * (qNei[IX] - qLoc[IX]);
					}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i, j, k;
						index2coord(index, i, j, k, params.isize, params.jsize, params.ksize);

						const int ghostWidth = params.ghostWidth;

						if (k>=ghostWidth-1 && k<=params.kmax-ghostWidth+1 &&
								j>=ghostWidth-1 && j<=params.jmax-ghostWidth+1 &&
								i>=ghostWidth-1 && i<=params.imax-ghostWidth+1)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j, k);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t phiLoc = phi(i, j, k);

							real_t uStarMinusX, piStarMinusX, Mmx;
							{
								const HydroState qMx = getHydroState(Qdata, i-1, j, k);
								const real_t cMinusX = computeSpeedSound(qMx);
								const real_t phiMx = phi(i-1, j, k);
								Mmx = computeM(qLoc, phiLoc, qMx, phiMx);
								computeAcousticRelaxation(qLoc, cLoc, qMx, cMinusX, Mmx, IU, -1,
										uStarMinusX, piStarMinusX);
							}

							real_t uStarPlusX, piStarPlusX, Mpx;
							{
								const HydroState qPx = getHydroState(Qdata, i+1, j, k);
								const real_t cPlusX = computeSpeedSound(qPx);
								const real_t phiPx = phi(i+1, j, k);
								Mpx = computeM(qLoc, phiLoc, qPx, phiPx);
								computeAcousticRelaxation(qLoc, cLoc, qPx, cPlusX, Mpx, IU, +1,
										uStarPlusX, piStarPlusX);
							}

							real_t uStarMinusY, piStarMinusY, Mmy;
							{
								const HydroState qMy = getHydroState(Qdata, i, j-1, k);
								const real_t cMinusY = computeSpeedSound(qMy);
								const real_t phiMy = phi(i, j-1, k);
								Mmy = computeM(qLoc, phiLoc, qMy, phiMy);
								computeAcousticRelaxation(qLoc, cLoc, qMy, cMinusY, Mmy, IV, -1,
										uStarMinusY, piStarMinusY);
							}

							real_t uStarPlusY, piStarPlusY, Mpy;
							{
								const HydroState qPy = getHydroState(Qdata, i, j+1, k);
								const real_t cPlusY = computeSpeedSound(qPy);
								const real_t phiPy = phi(i, j+1, k);
								Mpy = computeM(qLoc, phiLoc, qPy, phiPy);
								computeAcousticRelaxation(qLoc, cLoc, qPy, cPlusY, Mpy, IV, +1,
										uStarPlusY, piStarPlusY);
							}


							real_t uStarMinusZ, piStarMinusZ, Mmz;
							{
								const HydroState qMz = getHydroState(Qdata, i, j, k-1);
								const real_t cMinusZ = computeSpeedSound(qMz);
								const real_t phiMz = phi(i, j, k-1);
								Mmz = computeM(qLoc, phiLoc, qMz, phiMz);
								computeAcousticRelaxation(qLoc, cLoc, qMz, cMinusZ, Mmz, IW, -1,
										uStarMinusZ, piStarMinusZ);
							}

							real_t uStarPlusZ, piStarPlusZ, Mpz;
							{
								const HydroState qPz = getHydroState(Qdata, i, j, k+1);
								const real_t cPlusZ = computeSpeedSound(qPz);
								const real_t phiPz = phi(i, j, k+1);
								Mpz = computeM(qLoc, phiLoc, qPz, phiPz);
								computeAcousticRelaxation(qLoc, cLoc, qPz, cPlusZ, Mpz, IW, +1,
										uStarPlusZ, piStarPlusZ);
							}

							HydroState uLoc = getHydroState(Udata, i, j, k);

							if (conservative)
							{
								uLoc[IE] += qLoc[ID] * phiLoc;
							}

							// Acoustic update
							uLoc[IU] -= dtdx * (piStarPlusX - piStarMinusX);
							uLoc[IV] -= dtdy * (piStarPlusY - piStarMinusY);
							uLoc[IW] -= dtdz * (piStarPlusZ - piStarMinusZ);
							uLoc[IP] -= dtdx * (piStarMinusX * uStarMinusX + piStarPlusX * uStarPlusX);
							uLoc[IP] -= dtdy * (piStarMinusY * uStarMinusY + piStarPlusY * uStarPlusY);
							uLoc[IP] -= dtdz * (piStarMinusZ * uStarMinusZ + piStarPlusZ * uStarPlusZ);

							uLoc[IU] -= half_dtdx * (Mpx - Mmx);
							uLoc[IV] -= half_dtdy * (Mpy - Mmy);
							uLoc[IW] -= half_dtdz * (Mpz - Mmz);
							if (!conservative)
							{
								uLoc[IP] -= half_dtdx * (Mpx * uStarPlusX + Mmx * uStarMinusX);
								uLoc[IP] -= half_dtdy * (Mpy * uStarPlusY + Mmy * uStarMinusY);
								uLoc[IP] -= half_dtdz * (Mpz * uStarPlusZ + Mmz * uStarMinusZ);
							}

							// Compute L factor
							const real_t invL = ONE_F / (ONE_F +
									dtdx * (uStarMinusX + uStarPlusX) +
									dtdy * (uStarMinusY + uStarPlusY) +
									dtdz * (uStarMinusZ + uStarPlusZ));

							uLoc[ID] *= invL;
							uLoc[IP] *= invL;
							uLoc[IS] *= invL;
							uLoc[IU] *= invL;
							uLoc[IV] *= invL;
							uLoc[IW] *= invL;

							// Real update
							setHydroState(Udata, uLoc, i, j, k);
						}
					}

				const DataArray Udata;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				const real_t m_K;
				const real_t dtdx;
				const real_t dtdy;
				const real_t dtdz;
				const real_t half_dtdx;
				const real_t half_dtdy;
				const real_t half_dtdz;
				const bool conservative;
		}; // ComputeAcousticStepFunctor3D


		class ComputeTransportStepFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeTransportStepFunctor3D(HydroParams params_,
						DataArrayConst Udata_, DataArrayConst Qdata_, DataArray U2data_, DataArrayConst gradphi_, DataArrayConst UGdata_,
						real_t dt_) :
					HydroBaseFunctor3D(params_),
					Udata(Udata_), Qdata(Qdata_), U2data(U2data_), gradphi(gradphi_), UGdata(UGdata_),
					dtdx(dt_/params.dx), dtdy(dt_/params.dy), dtdz(dt_/params.dz),
					conservative(params.settings.conservative) {};

				static void apply(HydroParams params,
						DataArrayConst Udata, DataArrayConst Qdata, DataArray U2data, DataArrayConst gradphi, DataArrayConst UGdata,
						real_t dt, int nbCells)
				{
					ComputeTransportStepFunctor3D functor(params, Udata, Qdata, U2data, gradphi, UGdata, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void computeAcousticRelaxation(const HydroState& qLoc, real_t cLoc,
							const HydroState& qNei, real_t cNei,
							real_t M, int IX, int dir, real_t& uStar) const
					{
						const real_t a = params.settings.K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
						uStar = dir * HALF_F * (qNei[IX] + qLoc[IX]) - HALF_F * (qNei[IP] - qLoc[IP] + M) / a;
					}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j,k;
						index2coord(index, i, j, k, params.isize, params.jsize, params.ksize);

						const int ghostWidth = params.ghostWidth;

						if (k>=ghostWidth && k<=params.kmax-ghostWidth &&
								j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j, k);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t phiLoc = phi(i, j, k);

							real_t uStarMinusX;
							{
								const HydroState qMx = getHydroState(Qdata, i-1, j, k);
								const real_t cMinusX = computeSpeedSound(qMx);
								const real_t phiMx = phi(i-1, j, k);
								const real_t Mmx = computeM(qLoc, phiLoc, qMx, phiMx);
								computeAcousticRelaxation(qLoc, cLoc, qMx, cMinusX, Mmx, IU, -1,
										uStarMinusX);
							}

							real_t uStarPlusX;
							{
								const HydroState qPx = getHydroState(Qdata, i+1, j, k);
								const real_t cPlusX = computeSpeedSound(qPx);
								const real_t phiPx = phi(i+1, j, k);
								const real_t Mpx = computeM(qLoc, phiLoc, qPx, phiPx);
								computeAcousticRelaxation(qLoc, cLoc, qPx, cPlusX, Mpx, IU, +1,
										uStarPlusX);
							}

							real_t uStarMinusY;
							{
								const HydroState qMy = getHydroState(Qdata, i, j-1, k);
								const real_t cMinusY = computeSpeedSound(qMy);
								const real_t phiMy = phi(i, j-1, k);
								const real_t Mmy = computeM(qLoc, phiLoc, qMy, phiMy);
								computeAcousticRelaxation(qLoc, cLoc, qMy, cMinusY, Mmy, IV, -1,
										uStarMinusY);
							}

							real_t uStarPlusY;
							{
								const HydroState qPy = getHydroState(Qdata, i, j+1, k);
								const real_t cPlusY = computeSpeedSound(qPy);
								const real_t phiPy = phi(i, j+1, k);
								const real_t Mpy = computeM(qLoc, phiLoc, qPy, phiPy);
								computeAcousticRelaxation(qLoc, cLoc, qPy, cPlusY, Mpy, IV, +1,
										uStarPlusY);
							}

							real_t uStarMinusZ;
							{
								const HydroState qMz = getHydroState(Qdata, i, j, k-1);
								const real_t cMinusZ = computeSpeedSound(qMz);
								const real_t phiMz = phi(i, j, k-1);
								const real_t Mmz = computeM(qLoc, phiLoc, qMz, phiMz);
								computeAcousticRelaxation(qLoc, cLoc, qMz, cMinusZ, Mmz, IW, -1,
										uStarMinusZ);
							}

							real_t uStarPlusZ;
							{
								const HydroState qPz = getHydroState(Qdata, i, j, k+1);
								const real_t cPlusZ = computeSpeedSound(qPz);
								const real_t phiPz = phi(i, j, k+1);
								const real_t Mpz = computeM(qLoc, phiLoc, qPz, phiPz);
								computeAcousticRelaxation(qLoc, cLoc, qPz, cPlusZ, Mpz, IW, +1,
										uStarPlusZ);
							}

							const HydroState uLoc = getHydroState(Udata, i, j, k);
							HydroState u2Loc = getHydroState(Udata, i, j, k);

							u2Loc[ID] += dtdx * uLoc[ID] * (uStarMinusX + uStarPlusX);
							u2Loc[IU] += dtdx * uLoc[IU] * (uStarMinusX + uStarPlusX);
							u2Loc[IV] += dtdx * uLoc[IV] * (uStarMinusX + uStarPlusX);
							u2Loc[IW] += dtdx * uLoc[IW] * (uStarMinusX + uStarPlusX);
							u2Loc[IP] += dtdx * uLoc[IP] * (uStarMinusX + uStarPlusX);
							u2Loc[IS] += dtdx * uLoc[IS] * (uStarMinusX + uStarPlusX);

							u2Loc[ID] += dtdy * uLoc[ID] * (uStarMinusY + uStarPlusY);
							u2Loc[IU] += dtdy * uLoc[IU] * (uStarMinusY + uStarPlusY);
							u2Loc[IV] += dtdy * uLoc[IV] * (uStarMinusY + uStarPlusY);
							u2Loc[IW] += dtdy * uLoc[IW] * (uStarMinusY + uStarPlusY);
							u2Loc[IP] += dtdy * uLoc[IP] * (uStarMinusY + uStarPlusY);
							u2Loc[IS] += dtdy * uLoc[IS] * (uStarMinusY + uStarPlusY);

							u2Loc[ID] += dtdz * uLoc[ID] * (uStarMinusZ + uStarPlusZ);
							u2Loc[IU] += dtdz * uLoc[IU] * (uStarMinusZ + uStarPlusZ);
							u2Loc[IV] += dtdz * uLoc[IV] * (uStarMinusZ + uStarPlusZ);
							u2Loc[IW] += dtdz * uLoc[IW] * (uStarMinusZ + uStarPlusZ);
							u2Loc[IP] += dtdz * uLoc[IP] * (uStarMinusZ + uStarPlusZ);
							u2Loc[IS] += dtdz * uLoc[IS] * (uStarMinusZ + uStarPlusZ);

							{
								const int i0 = (uStarMinusX > ZERO_F) ? i : i - 1;
								const HydroState u0 = getHydroState(Udata, i0, j, k);
								u2Loc[ID] -= dtdx * u0[ID] * uStarMinusX;
								u2Loc[IU] -= dtdx * u0[IU] * uStarMinusX;
								u2Loc[IV] -= dtdx * u0[IV] * uStarMinusX;
								u2Loc[IW] -= dtdx * u0[IW] * uStarMinusX;
								u2Loc[IP] -= dtdx * u0[IP] * uStarMinusX;
								u2Loc[IS] -= dtdx * u0[IS] * uStarMinusX;
							}

							{
								const int i0= (uStarPlusX > ZERO_F) ? i : i + 1;
								const HydroState u0 = getHydroState(Udata, i0, j, k);
								u2Loc[ID] -= dtdx * u0[ID] * uStarPlusX;
								u2Loc[IU] -= dtdx * u0[IU] * uStarPlusX;
								u2Loc[IV] -= dtdx * u0[IV] * uStarPlusX;
								u2Loc[IW] -= dtdx * u0[IW] * uStarPlusX;
								u2Loc[IP] -= dtdx * u0[IP] * uStarPlusX;
								u2Loc[IS] -= dtdx * u0[IS] * uStarPlusX;
							}

							{
								const int j0 = (uStarMinusY > ZERO_F) ? j : j - 1;
								const HydroState u0 = getHydroState(Udata, i, j0, k);
								u2Loc[ID] -= dtdy * u0[ID] * uStarMinusY;
								u2Loc[IU] -= dtdy * u0[IU] * uStarMinusY;
								u2Loc[IV] -= dtdy * u0[IV] * uStarMinusY;
								u2Loc[IW] -= dtdy * u0[IW] * uStarMinusY;
								u2Loc[IP] -= dtdy * u0[IP] * uStarMinusY;
								u2Loc[IS] -= dtdy * u0[IS] * uStarMinusY;
							}

							{
								const int j0 = (uStarPlusY > ZERO_F) ? j : j + 1;
								const HydroState u0 = getHydroState(Udata, i, j0, k);
								u2Loc[ID] -= dtdy * u0[ID] * uStarPlusY;
								u2Loc[IU] -= dtdy * u0[IU] * uStarPlusY;
								u2Loc[IV] -= dtdy * u0[IV] * uStarPlusY;
								u2Loc[IW] -= dtdy * u0[IW] * uStarPlusY;
								u2Loc[IP] -= dtdy * u0[IP] * uStarPlusY;
								u2Loc[IS] -= dtdy * u0[IS] * uStarPlusY;
							}

							{
								const int k0 = (uStarMinusZ > ZERO_F) ? k : k - 1;
								const HydroState u0 = getHydroState(Udata, i, j, k0);
								u2Loc[ID] -= dtdz * u0[ID] * uStarMinusZ;
								u2Loc[IU] -= dtdz * u0[IU] * uStarMinusZ;
								u2Loc[IV] -= dtdz * u0[IV] * uStarMinusZ;
								u2Loc[IW] -= dtdz * u0[IW] * uStarMinusZ;
								u2Loc[IP] -= dtdz * u0[IP] * uStarMinusZ;
								u2Loc[IS] -= dtdz * u0[IS] * uStarMinusZ;
							}

							{
								const int k0 = (uStarPlusZ > ZERO_F) ? k : k + 1;
								const HydroState u0 = getHydroState(Udata, i, j, k0);
								u2Loc[ID] -= dtdz * u0[ID] * uStarPlusZ;
								u2Loc[IU] -= dtdz * u0[IU] * uStarPlusZ;
								u2Loc[IV] -= dtdz * u0[IV] * uStarPlusZ;
								u2Loc[IW] -= dtdz * u0[IW] * uStarPlusZ;
								u2Loc[IP] -= dtdz * u0[IP] * uStarPlusZ;
								u2Loc[IS] -= dtdz * u0[IS] * uStarPlusZ;
							}

							if (conservative)
							{
								u2Loc[IE] -= u2Loc[ID] * phiLoc;
							}

							setHydroState(U2data, u2Loc, i, j, k);
						}
					}

				const DataArrayConst Udata;
				const DataArrayConst Qdata;
				const DataArray U2data;
				const DataArrayConst gradphi;
				const DataArrayConst UGdata;
				const real_t dtdx;
				const real_t dtdy;
				const real_t dtdz;
				const bool conservative;
		}; // ComputeTransportStepFunctor3D

		class ComputeStateChangeFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeStateChangeFunctor3D(HydroParams params_, DataArray Udata_, DataArray Qdata_, DataArray gradphi_):
					HydroBaseFunctor3D(params_), Udata(Udata_), Qdata(Qdata_), gradphi(gradphi_){};
				static void apply(HydroParams params,
						DataArray Udata, DataArray Qdata, DataArray gradphi,
						int nbCells)
				{
					ComputeStateChangeFunctor3D functor(params, Udata, Qdata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						/*
						   const int ghostWidth = params.ghostWidth;
						   int i, j, k;
						   if (k>=ghostWidth && k<=params.kmax-ghostWidth &&
						   j>=ghostWidth && j<=params.jmax-ghostWidth &&
						   i>=ghostWidth && i<=params.imax-ghostWidth)
						   {

						   }
						   */
					}
				const DataArray Udata;
				const DataArray Qdata;
				const DataArray gradphi;
		};

		class ComputeViscosityStepFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeViscosityStepFunctor3D(HydroParams params_, DataArray Udata_, DataArrayConst Qdata_, real_t dt_):
					HydroBaseFunctor3D(params_), Udata(Udata_), Qdata(Qdata_), dt(dt_) {};

				static void apply(HydroParams params,
						DataArray Udata, DataArrayConst Qdata,
						real_t dt, int nbCells)
				{
					ComputeViscosityStepFunctor3D functor(params, Udata, Qdata, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{

						const int ghostWidth = params.ghostWidth;
						int i, j, k;
						index2coord(index, i, j, k, params.isize, params.jsize, params.ksize);

						if (k>=ghostWidth && k<=params.kmax-ghostWidth &&
								j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{

							const real_t mu = params.settings.mu0;
							const real_t lambda = ZERO_F;
							const real_t eta = lambda - TWO_F/(ONE_F+TWO_F) * mu;

							const real_t dx = params.dx;
							const real_t dy = params.dy;
							const real_t dz = params.dz;
							const real_t dtdx = dt / dx;
							const real_t dtdy = dt / dy;
							const real_t dtdz = dt / dz;

							// Face X-
							{
								const HydroState qMx   = getHydroState(Qdata, i-1, j  , k  );
								const HydroState qPx   = getHydroState(Qdata, i  , j  , k  );
								const HydroState qMxMy = getHydroState(Qdata, i-1, j-1, k  );
								const HydroState qMxPy = getHydroState(Qdata, i-1, j+1, k  );
								const HydroState qPxMy = getHydroState(Qdata, i  , j-1, k  );
								const HydroState qPxPy = getHydroState(Qdata, i  , j+1, k  );
								const HydroState qMxMz = getHydroState(Qdata, i-1, j  , k-1);
								const HydroState qMxPz = getHydroState(Qdata, i-1, j  , k+1);
								const HydroState qPxMz = getHydroState(Qdata, i  , j  , k-1);
								const HydroState qPxPz = getHydroState(Qdata, i  , j  , k+1);

								const real_t uFace = (qMx[IU] + qPx[IU]) / TWO_F;
								const real_t vFace = (qMx[IV] + qPx[IV]) / TWO_F;
								const real_t wFace = (qMx[IW] + qPx[IW]) / TWO_F;

								const real_t du_dx = (qPx[IU] - qMx[IU]) / dx;
								const real_t dv_dx = (qPx[IV] - qMx[IV]) / dx;
								const real_t dw_dx = (qPx[IW] - qMx[IW]) / dx;
								const real_t du_dy = (qPxPy[IU] + qMxPy[IU] - qPxMy[IU] - qMxMy[IU]) / (FOUR_F * dy);
								const real_t dv_dy = (qPxPy[IV] + qMxPy[IV] - qPxMy[IV] - qMxMy[IV]) / (FOUR_F * dy);
								const real_t du_dz = (qPxPz[IU] + qMxPz[IU] - qPxMz[IU] - qMxMz[IU]) / (FOUR_F * dz);
								const real_t dw_dz = (qPxPz[IW] + qMxPz[IW] - qPxMz[IW] - qMxMz[IW]) / (FOUR_F * dz);

								// Compute fluxes
								const real_t tau_xx = TWO_F * mu * du_dx + eta * (du_dx + dv_dy + dw_dz);
								const real_t tau_xy = mu * (du_dy + dv_dx);
								const real_t tau_xz = mu * (du_dz + dw_dx);

								// Update the right cell of the interface
								Udata(i, j, k, IU) += - dtdx * tau_xx;
								Udata(i, j, k, IV) += - dtdx * tau_xy;
								Udata(i, j, k, IW) += - dtdx * tau_xz;
								Udata(i, j, k, IP) += - dtdx * (uFace * tau_xx + vFace * tau_xy + wFace * tau_xz);
							}

							// Face X+
							{
								const HydroState qMx   = getHydroState(Qdata, i  , j  , k  );
								const HydroState qPx   = getHydroState(Qdata, i+1, j  , k  );
								const HydroState qMxMy = getHydroState(Qdata, i  , j-1, k  );
								const HydroState qMxPy = getHydroState(Qdata, i  , j+1, k  );
								const HydroState qPxMy = getHydroState(Qdata, i+1, j-1, k  );
								const HydroState qPxPy = getHydroState(Qdata, i+1, j+1, k  );
								const HydroState qMxMz = getHydroState(Qdata, i  , j  , k-1);
								const HydroState qMxPz = getHydroState(Qdata, i  , j  , k+1);
								const HydroState qPxMz = getHydroState(Qdata, i+1, j  , k-1);
								const HydroState qPxPz = getHydroState(Qdata, i+1, j  , k+1);

								const real_t uFace = (qMx[IU] + qPx[IU]) / TWO_F;
								const real_t vFace = (qMx[IV] + qPx[IV]) / TWO_F;
								const real_t wFace = (qMx[IW] + qPx[IW]) / TWO_F;

								const real_t du_dx = (qPx[IU] - qMx[IU]) / dx;
								const real_t dv_dx = (qPx[IV] - qMx[IV]) / dx;
								const real_t dw_dx = (qPx[IW] - qMx[IW]) / dx;
								const real_t du_dy = (qPxPy[IU] + qMxPy[IU] - qPxMy[IU] - qMxMy[IU]) / (FOUR_F * dy);
								const real_t dv_dy = (qPxPy[IV] + qMxPy[IV] - qPxMy[IV] - qMxMy[IV]) / (FOUR_F * dy);
								const real_t du_dz = (qPxPz[IU] + qMxPz[IU] - qPxMz[IU] - qMxMz[IU]) / (FOUR_F * dz);
								const real_t dw_dz = (qPxPz[IW] + qMxPz[IW] - qPxMz[IW] - qMxMz[IW]) / (FOUR_F * dz);

								// Compute fluxes
								const real_t tau_xx = TWO_F * mu * du_dx + eta * (du_dx + dv_dy + dw_dz);
								const real_t tau_xy = mu * (du_dy + dv_dx);
								const real_t tau_xz = mu * (du_dz + dw_dx);

								// Update the right cell of the interface
								Udata(i, j, k, IU) +=   dtdx * tau_xx;
								Udata(i, j, k, IV) +=   dtdx * tau_xy;
								Udata(i, j, k, IW) +=   dtdx * tau_xz;
								Udata(i, j, k, IP) +=   dtdx * (uFace * tau_xx + vFace * tau_xy + wFace * tau_xz);
							}

							// Face Y-
							{
								const HydroState qMy   = getHydroState(Qdata, i  , j-1, k  );
								const HydroState qPy   = getHydroState(Qdata, i  , j  , k  );
								const HydroState qMxMy = getHydroState(Qdata, i-1, j-1, k  );
								const HydroState qPxMy = getHydroState(Qdata, i+1, j-1, k  );
								const HydroState qMxPy = getHydroState(Qdata, i-1, j  , k  );
								const HydroState qPxPy = getHydroState(Qdata, i+1, j  , k  );
								const HydroState qMyMz = getHydroState(Qdata, i  , j-1, k-1);
								const HydroState qMyPz = getHydroState(Qdata, i  , j-1, k+1);
								const HydroState qPyMz = getHydroState(Qdata, i  , j  , k-1);
								const HydroState qPyPz = getHydroState(Qdata, i  , j  , k+1);

								const real_t uFace = (qMy[IU] + qPy[IU]) / TWO_F;
								const real_t vFace = (qMy[IV] + qPy[IV]) / TWO_F;
								const real_t wFace = (qMy[IW] + qPy[IW]) / TWO_F;

								const real_t du_dx = (qPxMy[IU] + qPxPy[IU] - qMxMy[IU] - qMxPy[IU]) / (FOUR_F * dx);
								const real_t dv_dx = (qPxMy[IV] + qPxPy[IV] - qMxMy[IV] - qMxPy[IV]) / (FOUR_F * dx);
								const real_t du_dy = (qPy[IU] - qMy[IU]) / dy;
								const real_t dv_dy = (qPy[IV] - qMy[IV]) / dy;
								const real_t dw_dy = (qPy[IW] - qMy[IW]) / dy;
								const real_t dv_dz = (qMyPz[IV] + qPyPz[IV] - qMyMz[IV] - qPyMz[IV]) / (FOUR_F * dz);
								const real_t dw_dz = (qMyPz[IW] + qPyPz[IW] - qMyMz[IW] - qPyMz[IW]) / (FOUR_F * dz);

								const real_t tau_yx = mu * (dv_dx + du_dy);
								const real_t tau_yy = TWO_F * mu * dv_dy + eta * (du_dx + dv_dy + dw_dz);
								const real_t tau_yz = mu * (dv_dz + dw_dy);

								// Update the up cell of the interface
								Udata(i, j, k, IU) += - dtdy * tau_yx;
								Udata(i, j, k, IV) += - dtdy * tau_yy;
								Udata(i, j, k, IW) += - dtdy * tau_yz;
								Udata(i, j, k, IP) += - dtdy * (uFace * tau_yx + vFace * tau_yy + wFace * tau_yz);
							}

							// Face Y+
							{
								const HydroState qMy   = getHydroState(Qdata, i  , j  , k  );
								const HydroState qPy   = getHydroState(Qdata, i  , j+1, k  );
								const HydroState qMxMy = getHydroState(Qdata, i-1, j  , k  );
								const HydroState qPxMy = getHydroState(Qdata, i+1, j  , k  );
								const HydroState qMxPy = getHydroState(Qdata, i-1, j+1, k  );
								const HydroState qPxPy = getHydroState(Qdata, i+1, j+1, k  );
								const HydroState qMyMz = getHydroState(Qdata, i  , j  , k-1);
								const HydroState qMyPz = getHydroState(Qdata, i  , j  , k+1);
								const HydroState qPyMz = getHydroState(Qdata, i  , j+1, k-1);
								const HydroState qPyPz = getHydroState(Qdata, i  , j+1, k+1);

								const real_t uFace = (qMy[IU] + qPy[IU]) / TWO_F;
								const real_t vFace = (qMy[IV] + qPy[IV]) / TWO_F;
								const real_t wFace = (qMy[IW] + qPy[IW]) / TWO_F;

								const real_t du_dx = (qPxMy[IU] + qPxPy[IU] - qMxMy[IU] - qMxPy[IU]) / (FOUR_F * dx);
								const real_t dv_dx = (qPxMy[IV] + qPxPy[IV] - qMxMy[IV] - qMxPy[IV]) / (FOUR_F * dx);
								const real_t du_dy = (qPy[IU] - qMy[IU]) / dy;
								const real_t dv_dy = (qPy[IV] - qMy[IV]) / dy;
								const real_t dw_dy = (qPy[IW] - qMy[IW]) / dy;
								const real_t dv_dz = (qMyPz[IV] + qPyPz[IV] - qMyMz[IV] - qPyMz[IV]) / (FOUR_F * dz);
								const real_t dw_dz = (qMyPz[IW] + qPyPz[IW] - qMyMz[IW] - qPyMz[IW]) / (FOUR_F * dz);

								const real_t tau_yx = mu * (dv_dx + du_dy);
								const real_t tau_yy = TWO_F * mu * dv_dy + eta * (du_dx + dv_dy + dw_dz);
								const real_t tau_yz = mu * (dv_dz + dw_dy);

								// Update the bottom cell of the interface
								Udata(i, j, k, IU) +=   dtdy * tau_yx;
								Udata(i, j, k, IV) +=   dtdy * tau_yy;
								Udata(i, j, k, IW) +=   dtdy * tau_yz;
								Udata(i, j, k, IP) +=   dtdy * (uFace * tau_yx + vFace * tau_yy + wFace * tau_yz);
							}

							// Face Z-
							{
								const HydroState qMz   = getHydroState(Qdata, i  , j  , k-1);
								const HydroState qPz   = getHydroState(Qdata, i  , j  , k  );
								const HydroState qMxMz = getHydroState(Qdata, i-1, j  , k-1);
								const HydroState qPxMz = getHydroState(Qdata, i+1, j  , k-1);
								const HydroState qMxPz = getHydroState(Qdata, i-1, j  , k  );
								const HydroState qPxPz = getHydroState(Qdata, i+1, j  , k  );
								const HydroState qMyMz = getHydroState(Qdata, i  , j-1, k-1);
								const HydroState qPyMz = getHydroState(Qdata, i  , j+1, k-1);
								const HydroState qMyPz = getHydroState(Qdata, i  , j-1, k  );
								const HydroState qPyPz = getHydroState(Qdata, i  , j+1, k  );

								const real_t uFace = (qMz[IU] + qPz[IU]) / TWO_F;
								const real_t vFace = (qMz[IV] + qPz[IV]) / TWO_F;
								const real_t wFace = (qMz[IW] + qPz[IW]) / TWO_F;

								const real_t du_dx = (qPxPz[IU] + qPxMz[IU] - qMxPz[IU] - qMxMz[IU]) / (FOUR_F * dx);
								const real_t dw_dx = (qPxPz[IW] + qPxMz[IW] - qMxPz[IW] - qMxMz[IW]) / (FOUR_F * dx);
								const real_t dv_dy = (qPyPz[IV] + qPyMz[IV] - qMyPz[IV] - qMyMz[IV]) / (FOUR_F * dy);
								const real_t dw_dy = (qPyPz[IW] + qPyMz[IW] - qMyPz[IW] - qMyMz[IW]) / (FOUR_F * dy);
								const real_t du_dz = (qPz[IU] - qMz[IU]) / dz;
								const real_t dv_dz = (qPz[IV] - qMz[IV]) / dz;
								const real_t dw_dz = (qPz[IW] - qMz[IW]) / dz;

								const real_t tau_zx = mu * (dw_dx + du_dz);
								const real_t tau_zy = mu * (dw_dy + dv_dz);
								const real_t tau_zz = TWO_F * mu * dw_dz + eta * (du_dx + dv_dy + dw_dz);

								// Update the bottom cell of the interface
								Udata(i, j, k, IU) += - dtdz * tau_zx;
								Udata(i, j, k, IV) += - dtdz * tau_zy;
								Udata(i, j, k, IW) += - dtdz * tau_zz;
								Udata(i, j, k, IP) += - dtdz * (uFace * tau_zx + vFace * tau_zy + wFace * tau_zz);
							}

							// Face Z+
							{
								const HydroState qMz   = getHydroState(Qdata, i  , j  , k  );
								const HydroState qPz   = getHydroState(Qdata, i  , j  , k+1);
								const HydroState qMxMz = getHydroState(Qdata, i-1, j  , k  );
								const HydroState qPxMz = getHydroState(Qdata, i+1, j  , k  );
								const HydroState qMxPz = getHydroState(Qdata, i-1, j  , k+1);
								const HydroState qPxPz = getHydroState(Qdata, i+1, j  , k+1);
								const HydroState qMyMz = getHydroState(Qdata, i  , j-1, k  );
								const HydroState qPyMz = getHydroState(Qdata, i  , j+1, k  );
								const HydroState qMyPz = getHydroState(Qdata, i  , j-1, k+1);
								const HydroState qPyPz = getHydroState(Qdata, i  , j+1, k+1);

								const real_t uFace = (qMz[IU] + qPz[IU]) / TWO_F;
								const real_t vFace = (qMz[IV] + qPz[IV]) / TWO_F;
								const real_t wFace = (qMz[IW] + qPz[IW]) / TWO_F;

								const real_t du_dx = (qPxPz[IU] + qPxMz[IU] - qMxPz[IU] - qMxMz[IU]) / (FOUR_F * dx);
								const real_t dw_dx = (qPxPz[IW] + qPxMz[IW] - qMxPz[IW] - qMxMz[IW]) / (FOUR_F * dx);
								const real_t dv_dy = (qPyPz[IV] + qPyMz[IV] - qMyPz[IV] - qMyMz[IV]) / (FOUR_F * dy);
								const real_t dw_dy = (qPyPz[IW] + qPyMz[IW] - qMyPz[IW] - qMyMz[IW]) / (FOUR_F * dy);
								const real_t du_dz = (qPz[IU] - qMz[IU]) / dz;
								const real_t dv_dz = (qPz[IV] - qMz[IV]) / dz;
								const real_t dw_dz = (qPz[IW] - qMz[IW]) / dz;

								const real_t tau_zx = mu * (dw_dx + du_dz);
								const real_t tau_zy = mu * (dw_dy + dv_dz);
								const real_t tau_zz = TWO_F * mu * dw_dz + eta * (du_dx + dv_dy + dw_dz);

								// Update the bottom cell of the interface
								Udata(i, j, k, IU) +=   dtdz * tau_zx;
								Udata(i, j, k, IV) +=   dtdz * tau_zy;
								Udata(i, j, k, IW) +=   dtdz * tau_zz;
								Udata(i, j, k, IP) +=   dtdz * (uFace * tau_zx + vFace * tau_zy + wFace * tau_zz);
							}
						}
					}

				const DataArray Udata;
				const DataArrayConst Qdata;
				const real_t dt;
		}; // ComputeViscosityStepFunctor3D


		class ComputeHeatDiffusionStepFunctor3D : HydroBaseFunctor3D
		{
			public:
				ComputeHeatDiffusionStepFunctor3D(HydroParams params_, DataArray Udata_, DataArrayConst Qdata_, real_t dt_):
					HydroBaseFunctor3D(params_), Udata(Udata_), Qdata(Qdata_), dt(dt_) {};

				static void apply(HydroParams params,
						DataArray Udata, DataArrayConst Qdata,
						real_t dt, int nbCells)
				{
					ComputeHeatDiffusionStepFunctor3D functor(params, Udata, Qdata, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{

						const int ghostWidth = params.ghostWidth;
						const real_t kappa = params.settings.kappa0;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t dz = params.dz;
						const real_t dtdx = dt / dx;
						const real_t dtdy = dt / dy;
						const real_t dtdz = dt / dz;

						int i, j, k;
						index2coord(index, i, j, k, params.isize, params.jsize, params.ksize);

						if (k>=ghostWidth && k<=params.kmax-ghostWidth &&
								j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j, k);
							const real_t TLoc = computeTemperature(qLoc);
							real_t energy_fluxes = ZERO_F;

							{
								const HydroState qNei = getHydroState(Qdata, i-1, j, k);
								const real_t TNei = computeTemperature(qNei);
								energy_fluxes += dtdx * kappa * (TNei - TLoc) / dx;
							}

							{
								const HydroState qNei = getHydroState(Qdata, i+1, j, k);
								const real_t TNei = computeTemperature(qNei);
								energy_fluxes += dtdx * kappa * (TNei - TLoc) / dx;
							}

							{
								const HydroState qNei = getHydroState(Qdata, i, j-1, k);
								const real_t TNei = computeTemperature(qNei);
								energy_fluxes += dtdy * kappa * (TNei - TLoc) / dy;
							}

							{
								const HydroState qNei = getHydroState(Qdata, i, j+1, k);
								const real_t TNei = computeTemperature(qNei);
								energy_fluxes += dtdy * kappa * (TNei - TLoc) / dy;
							}

							{
								const HydroState qNei = getHydroState(Qdata, i, j, k-1);
								const real_t TNei = computeTemperature(qNei);
								energy_fluxes += dtdz * kappa * (TNei - TLoc) / dz;
							}

							{
								const HydroState qNei = getHydroState(Qdata, i, j, k+1);
								const real_t TNei = computeTemperature(qNei);
								energy_fluxes += dtdz * kappa * (TNei - TLoc) / dz;
							}

							Udata(i, j, k, IP) += energy_fluxes;
						}
					}

				const DataArray Udata;
				const DataArrayConst Qdata;
				const real_t dt;
		}; // ComputeHeatDiffusionStepFunctor3D


		class ComputeDtFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeDtFunctor3D(HydroParams params, DataArrayConst Udata) :
					HydroBaseFunctor3D(params), Udata(Udata)  {};

				static void apply(HydroParams params, DataArrayConst Udata, real_t& invDt, int nbCells)
				{
					ComputeDtFunctor3D functor(params, Udata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}

				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						// The identity under max is -Inf.
						// Kokkos does not come with a portable way to access
						// floating-point Inf and NaN.
#ifdef __CUDA_ARCH__
						dst = -CUDART_INF;
#else
						dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ksize = params.ksize;
						const int ghostWidth = params.ghostWidth;

						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t dz = params.dz;

						int i,j,k;
						index2coord(index,i,j,k,isize,jsize,ksize);

						if(k >= ghostWidth && k < ksize - ghostWidth &&
								j >= ghostWidth && j < jsize - ghostWidth &&
								i >= ghostWidth && i < isize - ghostWidth)
						{
							// get local conservative variable
							const HydroState uLoc = getHydroState(Udata, i, j, k);
							// get primitive variables in current cell
							const HydroState qLoc = computePrimitives(uLoc);
							const real_t c = computeSpeedSound(qLoc);
							const real_t vx = c+FABS(qLoc[IU]);
							const real_t vy = c+FABS(qLoc[IV]);
							const real_t vz = c+FABS(qLoc[IW]);

							// Hyperbolic part
							invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);
							// Viscous flux
							// (1.0*mu+abs(-2.0/3.0*mu)=8.0/3.0*mu
							//  formula still needs some justification)
							invDt = FMAX(invDt, 8.0 / 3.0 * params.settings.mu0 / uLoc[ID] * (ONE_F/(dx*dx) + ONE_F/(dy*dy) + ONE_F/(dz*dz)));
							// Heat flux
							invDt = FMAX(invDt, params.settings.kappa0 / (uLoc[ID] * params.settings.cp0) * (ONE_F/(dx*dx) + ONE_F/(dy*dy) + ONE_F/(dz*dz)));
						}
					} // operator ()


				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src) {
							dst = src;
						}
					} // join

				const DataArrayConst Udata;
		}; // ComputeDtFunctor3D


		class ComputeAcousticDtFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeAcousticDtFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeAcousticDtFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}

				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ksize = params.ksize;
						const int ghostWidth = params.ghostWidth;

						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t dz = params.dz;

						int i, j, k;
						index2coord(index, i, j, k, isize, jsize, ksize);

						if(k >= ghostWidth && k <= ksize - ghostWidth &&
								j >= ghostWidth && j <= jsize - ghostWidth &&
								i >= ghostWidth && i <= isize - ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j, k);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t K    = params.settings.K;

							if (j != jsize-ghostWidth && k != ksize-ghostWidth)
							{
								const HydroState qNei = getHydroState(Qdata, i-1, j, k);
								const real_t cNei = computeSpeedSound(qNei);

								const real_t aNei = K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
								const real_t invDtNei = aNei / FMIN(dx * qNei[ID], dx * qLoc[ID]);
								invDt = FMAX(invDt, invDtNei);
							}

							if (i != isize-ghostWidth && k != ksize-ghostWidth)
							{
								const HydroState qNei = getHydroState(Qdata, i, j-1, k);
								const real_t cNei = computeSpeedSound(qNei);

								const real_t aNei = K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
								const real_t invDtNei = aNei / FMIN(dy * qNei[ID], dy * qLoc[ID]);
								invDt = FMAX(invDt, invDtNei);
							}

							if (i != isize-ghostWidth && j != jsize-ghostWidth)
							{
								const HydroState qNei = getHydroState(Qdata, i, j, k-1);
								const real_t cNei = computeSpeedSound(qNei);

								const real_t aNei = K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
								const real_t invDtNei = aNei / FMIN(dz * qNei[ID], dz * qLoc[ID]);
								invDt = FMAX(invDt, invDtNei);
							}
						}
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeAcousticDtFunctor3D


		class ComputeTransportDtFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeTransportDtFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeTransportDtFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}

				KOKKOS_INLINE_FUNCTION
					void computeAcousticRelaxation(const HydroState& qLoc, real_t cLoc,
							const HydroState& qNei, real_t cNei,
							real_t M, int IX, int dir, real_t& uStar) const
					{
						const real_t a = params.settings.K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
						uStar = dir * HALF_F * (qNei[IX] + qLoc[IX]) - HALF_F * (qNei[IP] - qLoc[IP] + M) / a;
					}

				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ksize = params.ksize;
						const int ghostWidth = params.ghostWidth;

						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t dz = params.dz;

						int i, j, k;
						index2coord(index, i, j, k, isize, jsize, ksize);

						if(k >= ghostWidth && k <= ksize - ghostWidth &&
								j >= ghostWidth && j <= jsize - ghostWidth &&
								i >= ghostWidth && i <= isize - ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j, k);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t phiLoc = phi(i, j, k);

							real_t invDtLoc = ZERO_F;

							{
								const HydroState qMx = getHydroState(Qdata, i-1, j, k);
								const real_t cMinusX = computeSpeedSound(qMx);
								const real_t phiMx = phi(i-1, j, k);
								const real_t Mmx = computeM(qLoc, phiLoc, qMx, phiMx);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qMx, cMinusX, Mmx, IU, -1,
										uStar);
								invDtLoc += FABS(uStar) / dx;
							}

							{
								const HydroState qPx = getHydroState(Qdata, i+1, j, k);
								const real_t cPlusX = computeSpeedSound(qPx);
								const real_t phiPx = phi(i+1, j, k);
								const real_t Mpx = computeM(qLoc, phiLoc, qPx, phiPx);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qPx, cPlusX, Mpx, IU, +1,
										uStar);
								invDtLoc += FABS(uStar) / dx;
							}

							{
								const HydroState qMy = getHydroState(Qdata, i, j-1, k);
								const real_t cMinusY = computeSpeedSound(qMy);
								const real_t phiMy = phi(i, j-1, k);
								const real_t Mmy = computeM(qLoc, phiLoc, qMy, phiMy);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qMy, cMinusY, Mmy, IV, -1,
										uStar);
								invDtLoc += FABS(uStar) / dy;
							}

							{
								const HydroState qPy = getHydroState(Qdata, i, j+1, k);
								const real_t cPlusY = computeSpeedSound(qPy);
								const real_t phiPy = phi(i, j+1, k);
								const real_t Mpy = computeM(qLoc, phiLoc, qPy, phiPy);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qPy, cPlusY, Mpy, IV, +1,
										uStar);
								invDtLoc += FABS(uStar) / dy;
							}

							{
								const HydroState qMz = getHydroState(Qdata, i, j, k-1);
								const real_t cMinusZ = computeSpeedSound(qMz);
								const real_t phiMz = phi(i, j, k-1);
								const real_t Mmz = computeM(qLoc, phiLoc, qMz, phiMz);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qMz, cMinusZ, Mmz, IW, -1,
										uStar);
								invDtLoc += FABS(uStar) / dz;
							}

							{
								const HydroState qPz = getHydroState(Qdata, i, j, k+1);
								const real_t cPlusZ = computeSpeedSound(qPz);
								const real_t phiPz = phi(i, j, k+1);
								const real_t Mpz = computeM(qLoc, phiLoc, qPz, phiPz);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qPz, cPlusZ, Mpz, IW, +1,
										uStar);
								invDtLoc += FABS(uStar) / dz;
							}

							invDt = FMAX(invDt, invDtLoc);
						}
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D


		class ConvertToPrimitivesFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ConvertToPrimitivesFunctor3D(HydroParams params, DataArrayConst Udata, DataArray Qdata) :
					HydroBaseFunctor3D(params), Udata(Udata), Qdata(Qdata)  {};

				static void apply(HydroParams params,
						DataArrayConst Udata, DataArray Qdata,
						int nbCells)
				{
					ConvertToPrimitivesFunctor3D functor(params, Udata, Qdata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ksize = params.ksize;
						//const int ghostWidth = params.ghostWidth;

						int i,j,k;
						index2coord(index,i,j,k,isize,jsize,ksize);

						if(k >= 0 && k < ksize  &&
								j >= 0 && j < jsize  &&
								i >= 0 && i < isize )
						{
							// get local conservative variable
							const HydroState uLoc = getHydroState(Udata, i, j, k);
							// get primitive variables in current cell
							const HydroState qLoc = computePrimitives(uLoc);
							// copy q state in q global
							setHydroState(Qdata, qLoc, i, j, k);
						}
					}

				const DataArrayConst Udata;
				const DataArray Qdata;
		}; // ConvertToPrimitivesFunctor3D
		class CopyGradientFunctor3D : HydroBaseFunctor3D
		{
			public:
				CopyGradientFunctor3D(HydroParams params_,
						DataArray Qdata_,const int KX_, DataArray Udata_, const int KY_) :
					HydroBaseFunctor3D(params_),
					Qdata(Qdata_), KX(KX_), Udata(Udata_), KY(KY_) {};

				static void apply(HydroParams params,
						DataArray Qdata, const int KX, DataArray Udata, const int KY,
						int nbCells)
				{
					CopyGradientFunctor3D functor(params, Qdata, KX, Udata, KY);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
					}

				const DataArray Qdata;
				const int KX;
				const DataArray Udata;
				const int KY;
				const DataArray gradphi;
		}; 

	} // namespace all_regime

} // namespace euler_kokkos
