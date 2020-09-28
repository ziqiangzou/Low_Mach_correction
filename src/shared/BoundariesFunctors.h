#ifndef BOUNDARIES_FUNCTORS_H_
#define BOUNDARIES_FUNCTORS_H_

#include "HydroParams.h"    // for HydroParams
#include "kokkos_shared.h"  // for Data arrays

/**
 * Functors to update ghost cells (Hydro 2D).
 *
 */
template <FaceIdType faceId>
class MakeBoundariesLSCFunctor2D
{
	public:
		MakeBoundariesLSCFunctor2D(HydroParams params,
				DataArray2d Udata) :
			params(params), Udata(Udata) {};

		static void apply(HydroParams params, DataArray2d Udata, int nbCells)
		{
			MakeBoundariesLSCFunctor2D<faceId> functor(params, Udata);
			Kokkos::parallel_for(nbCells, functor);
		}

		KOKKOS_INLINE_FUNCTION
			void operator()(const int& index) const
			{
				const int nx = params.nx;
				const int ny = params.ny;

				const int ghostWidth = params.ghostWidth;

				const int imin = params.imin;
				const int imax = params.imax;


				int i,j;

				int boundary_type;

				int i0, j0;

				if (faceId == FACE_XMIN)
				{
					// boundary xmin
					boundary_type = params.boundary_type_xmin;

					j = index / ghostWidth;
					i = index - j*ghostWidth;

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= 0    && i <ghostWidth)
					{
						if (boundary_type==BC_NEUMANN||boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							const real_t phiBoundary=Udata(ghostWidth, j, IH);
							const real_t phiBoundaryp=Udata(ghostWidth+1, j, IH);
							const real_t delta_phi=phiBoundaryp - phiBoundary;
							Udata(i, j, IH)=phiBoundary+(i-ghostWidth)*delta_phi;
						}
						else
						{
							i0=nx+i;
							Udata(i  ,j  , IH) = Udata(i0  ,j  , IH);

						}
					}
				} // end FACE_XMIN

				if (faceId == FACE_XMAX)
				{
					// boundary xmax
					boundary_type = params.boundary_type_xmax;

					j = index / ghostWidth;
					i = index - j*ghostWidth;
					i += (nx+ghostWidth);

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= nx+ghostWidth && i <= nx+2*ghostWidth-1)
					{
						if (boundary_type==BC_NEUMANN||boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							const real_t phiBoundary=Udata(nx+ghostWidth-1, j, IH);
							const real_t phiBoundaryp=Udata(ghostWidth+nx-2, j, IH);
							const real_t delta_phi=phiBoundary - phiBoundaryp;
							Udata(i, j, IH)=phiBoundary+(i-ghostWidth-nx+1)*delta_phi;
						}
						else
						{
							i0=i-nx;
							Udata(i  ,j  , IH) = Udata(i0  ,j  , IH);

						}
					}
				} // end FACE_XMAX

				if (faceId == FACE_YMIN)
				{
					// boundary ymin
					boundary_type = params.boundary_type_ymin;

					i = index / ghostWidth;
					j = index - i*ghostWidth;

					if(i >= imin && i <= imax    &&
							j >= 0    && j <ghostWidth)
					{

						if (boundary_type==BC_NEUMANN||boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							const real_t phiBoundary=Udata(i, ghostWidth, IH);
							const real_t phiBoundaryp=Udata(i, ghostWidth+1, IH);
							const real_t delta_phi=phiBoundaryp - phiBoundary;
							Udata(i, j, IH)=phiBoundary+(j-ghostWidth)*delta_phi;
						}
						else
						{
							j0=ny+j;
							Udata(i  ,j  , IH) = Udata(i  ,j0  , IH);

						}
					}
				} // end FACE_YMIN

				if (faceId == FACE_YMAX)
				{
					// boundary ymax
					boundary_type = params.boundary_type_ymax;

					i = index / ghostWidth;
					j = index - i*ghostWidth;
					j += (ny+ghostWidth);
					if(i >= imin          && i <= imax              &&
							j >= ny+ghostWidth && j <= ny+2*ghostWidth-1)
					{

						if (boundary_type==BC_NEUMANN||boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							const real_t phiBoundary=Udata(i, ny+ghostWidth-1, IH);
							const real_t phiBoundaryp=Udata(i, ghostWidth+ny-2, IH);
							const real_t delta_phi=phiBoundary - phiBoundaryp;
							Udata(i, j, IH)=phiBoundary+(j-ghostWidth-ny+1)*delta_phi;
						}
						else
						{
							j0=j-ny;
							Udata(i  ,j  , IH) = Udata(i  ,j0  , IH);

						}
					}
				} // end FACE_YMAX
			} // end operator ()

		HydroParams params;
		DataArray2d Udata;
}; // MakeBoundariesLSFunctor2D
template <FaceIdType faceId>
class MakeBoundariesLSFunctor2D
{
	public:
		MakeBoundariesLSFunctor2D(HydroParams params,
				DataArray2d Udata) :
			params(params), Udata(Udata) {};

		static void apply(HydroParams params, DataArray2d Udata, int nbCells)
		{
			MakeBoundariesLSFunctor2D<faceId> functor(params, Udata);
			Kokkos::parallel_for(nbCells, functor);
		}

		KOKKOS_INLINE_FUNCTION
			void operator()(const int& index) const
			{
				const int nx = params.nx;
				const int ny = params.ny;

				const int ghostWidth = params.ghostWidth;

				const int imin = params.imin;
				const int imax = params.imax;


				int i,j;

				int boundary_type;

				int i0, j0;

				if (faceId == FACE_XMIN)
				{
					// boundary xmin
					boundary_type = params.boundary_type_xmin;

					j = index / ghostWidth;
					i = index - j*ghostWidth;

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= 0    && i <ghostWidth)
					{
						if (boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							i0=2*ghostWidth-1-i;
							Udata(i  ,j  , IH) = Udata(i0  ,j  , IH);

						}
						else if (boundary_type==BC_NEUMANN)
						{
							const real_t phiBoundary=Udata(ghostWidth, j, IH);
							const real_t phiBoundaryp=Udata(ghostWidth+1, j, IH);
							const real_t delta_phi=phiBoundaryp - phiBoundary;
							Udata(i, j, IH)=phiBoundary+(i-ghostWidth)*delta_phi;
						}
						else
						{
							i0=nx+i;
							Udata(i  ,j  , IH) = Udata(i0  ,j  , IH);

						}
					}
				} // end FACE_XMIN

				if (faceId == FACE_XMAX)
				{
					// boundary xmax
					boundary_type = params.boundary_type_xmax;

					j = index / ghostWidth;
					i = index - j*ghostWidth;
					i += (nx+ghostWidth);

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= nx+ghostWidth && i <= nx+2*ghostWidth-1)
					{
						if (boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							i0=2*nx+2*ghostWidth-1-i;
							Udata(i  ,j  , IH) = Udata(i0  ,j  , IH);

						}
						else if (boundary_type==BC_NEUMANN)
						{
							const real_t phiBoundary=Udata(nx+ghostWidth-1, j, IH);
							const real_t phiBoundaryp=Udata(ghostWidth+nx-2, j, IH);
							const real_t delta_phi=phiBoundary - phiBoundaryp;
							Udata(i, j, IH)=phiBoundary+(i-ghostWidth-nx+1)*delta_phi;
						}
						else
						{
							i0=i-nx;
							Udata(i  ,j  , IH) = Udata(i0  ,j  , IH);

						}
					}
				} // end FACE_XMAX

				if (faceId == FACE_YMIN)
				{
					// boundary ymin
					boundary_type = params.boundary_type_ymin;

					i = index / ghostWidth;
					j = index - i*ghostWidth;

					if(i >= imin && i <= imax    &&
							j >= 0    && j <ghostWidth)
					{

						if (boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							j0=2*ghostWidth-1-j;
							Udata(i  ,j  , IH) = Udata(i  ,j0  , IH);

						}
						else if (boundary_type==BC_NEUMANN)
						{
							const real_t phiBoundary=Udata(i, ghostWidth, IH);
							const real_t phiBoundaryp=Udata(i, ghostWidth+1, IH);
							const real_t delta_phi=phiBoundaryp - phiBoundary;
							Udata(i, j, IH)=phiBoundary+(j-ghostWidth)*delta_phi;
						}
						else
						{
							j0=ny+j;
							Udata(i  ,j  , IH) = Udata(i  ,j0  , IH);

						}
					}
				} // end FACE_YMIN

				if (faceId == FACE_YMAX)
				{
					// boundary ymax
					boundary_type = params.boundary_type_ymax;

					i = index / ghostWidth;
					j = index - i*ghostWidth;
					j += (ny+ghostWidth);
					if(i >= imin          && i <= imax              &&
							j >= ny+ghostWidth && j <= ny+2*ghostWidth-1)
					{

						if (boundary_type==BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED)
						{
							j0=2*ny+2*ghostWidth-1-j;
							Udata(i  ,j  , IH) = Udata(i  ,j0  , IH);

						}
						else if (boundary_type==BC_NEUMANN)
						{
							const real_t phiBoundary=Udata(i, ny+ghostWidth-1, IH);
							const real_t phiBoundaryp=Udata(i, ghostWidth+ny-2, IH);
							const real_t delta_phi=phiBoundary - phiBoundaryp;
							Udata(i, j, IH)=phiBoundary+(j-ghostWidth-ny+1)*delta_phi;
						}
						else
						{
							j0=j-ny;
							Udata(i  ,j  , IH) = Udata(i  ,j0  , IH);

						}
					}
				} // end FACE_YMAX
			} // end operator ()

		HydroParams params;
		DataArray2d Udata;
}; // MakeBoundariesLSFunctor2D

template <FaceIdType faceId>
class MakeBoundariesGradPhiFunctor2D
{
	public:
		MakeBoundariesGradPhiFunctor2D(HydroParams params,
				DataArray2d Udata) :
			params(params), Udata(Udata) {};

		static void apply(HydroParams params, DataArray2d Udata, int nbCells)
		{
			MakeBoundariesGradPhiFunctor2D<faceId> functor(params, Udata);
			Kokkos::parallel_for(nbCells, functor);
		}

		KOKKOS_INLINE_FUNCTION
			void operator()(const int& index) const
			{
				const int nx = params.nx;
				const int ny = params.ny;

				const int ghostWidth = params.ghostWidth;

				const int imin = params.imin;
				const int imax = params.imax;


				int i,j;

				int boundary_type;

				int i0, j0;
				int iVar;

				if (faceId == FACE_XMIN)
				{
					// boundary xmin
					boundary_type = params.boundary_type_xmin;

					j = index / ghostWidth;
					i = index - j*ghostWidth;

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= 0    && i <ghostWidth)
					{
						for ( iVar=0; iVar<4; iVar++ )
						{
							real_t sign=1.0;
							if ( boundary_type == BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED )
							{
								i0=2*ghostWidth-1-i;
								if (iVar==IPX) sign=-ONE_F;
							}
							else if( boundary_type == BC_NEUMANN )
							{
								i0=ghostWidth;
							}
							else
							{
								// periodic
								i0=nx+i;
							}

							Udata(i  ,j  , iVar) = Udata(i0  ,j  , iVar)*sign;
						}
					}
				} // end FACE_XMIN

				if (faceId == FACE_XMAX)
				{
					// boundary xmax
					boundary_type = params.boundary_type_xmax;

					j = index / ghostWidth;
					i = index - j*ghostWidth;
					i += (nx+ghostWidth);

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= nx+ghostWidth && i <= nx+2*ghostWidth-1)
					{
						for ( iVar=0; iVar<4; iVar++ )
						{
							real_t sign=1.0;
							if ( boundary_type == BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED  )
							{
								i0=2*nx+2*ghostWidth-1-i;
								if (iVar==IPX) sign=-ONE_F;
							} else if ( boundary_type == BC_NEUMANN )
							{
								i0=nx+ghostWidth-1;
							}
							else
							{
								// periodic
								i0=i-nx;
							}

							Udata(i  ,j  , iVar) = Udata(i0 ,j  , iVar)*sign;
						}
					}
				} // end FACE_XMAX

				if (faceId == FACE_YMIN)
				{
					// boundary ymin
					boundary_type = params.boundary_type_ymin;

					i = index / ghostWidth;
					j = index - i*ghostWidth;

					if(i >= imin && i <= imax    &&
							j >= 0    && j <ghostWidth)
					{

						for ( iVar=0; iVar<4; iVar++ )
						{
							real_t sign=1.0;
							if ( boundary_type == BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED  )
							{
								j0=2*ghostWidth-1-j;
								if (iVar==IPY) sign=-ONE_F;
							}
							else if ( boundary_type == BC_NEUMANN )
							{
								j0=ghostWidth;
							}
							else
							{
								// periodic
								j0=ny+j;
							}

							Udata(i  ,j  , iVar) = Udata(i  ,j0 , iVar)*sign;
						}
					}
				} // end FACE_YMIN

				if (faceId == FACE_YMAX)
				{
					// boundary ymax
					boundary_type = params.boundary_type_ymax;

					i = index / ghostWidth;
					j = index - i*ghostWidth;
					j += (ny+ghostWidth);
					if(i >= imin          && i <= imax              &&
							j >= ny+ghostWidth && j <= ny+2*ghostWidth-1)
					{

						for ( iVar=0; iVar<4; iVar++ )
						{
							real_t sign=1.0;
							if ( boundary_type == BC_DIRICHLET||boundary_type==BC_PROBLEM_DEFINED  )
							{
								j0=2*ny+2*ghostWidth-1-j;
								if (iVar==IPY) sign=-ONE_F;
							}
							else if ( boundary_type == BC_NEUMANN )
							{
								j0=ny+ghostWidth-1;
							}
							else
							{
								// periodic
								j0=j-ny;
							}

							Udata(i  ,j  , iVar) = Udata(i  ,j0  , iVar)*sign;
						}
					}
				} // end FACE_YMAX
			} // end operator ()

		HydroParams params;
		DataArray2d Udata;
}; // MakeBoundariesFunctor2D
template <FaceIdType faceId>
class MakeBoundariesFunctor2D
{
	public:
		MakeBoundariesFunctor2D(HydroParams params,
				DataArray2d Udata) :
			params(params), Udata(Udata) {};

		static void apply(HydroParams params, DataArray2d Udata, int nbCells)
		{
			MakeBoundariesFunctor2D<faceId> functor(params, Udata);
			Kokkos::parallel_for(nbCells, functor);
		}

		KOKKOS_INLINE_FUNCTION
			void operator()(const int& index) const
			{
				const int nx = params.nx;
				const int ny = params.ny;

				const int ghostWidth = params.ghostWidth;
				const int nbvar = params.nbvar;

				const int imin = params.imin;
				const int imax = params.imax;


				int i,j;

				int boundary_type;

				int i0, j0;
				int iVar;

				if (faceId == FACE_XMIN)
				{
					// boundary xmin
					boundary_type = params.boundary_type_xmin;

					j = index / ghostWidth;
					i = index - j*ghostWidth;

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= 0    && i <ghostWidth)
					{
						real_t sign=1.0;
						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if (iVar!=IH)
							{
								if ( boundary_type == BC_DIRICHLET )
								{
									i0=2*ghostWidth-1-i;
									if (iVar==IU) sign=-ONE_F;
								}
								else if( boundary_type == BC_NEUMANN )
								{
									i0=ghostWidth;
								}
								else
								{
									// periodic
									i0=nx+i;
								}

								Udata(i  ,j  , iVar) = Udata(i0  ,j  , iVar)*sign;
							}
						}
					}
				} // end FACE_XMIN

				if (faceId == FACE_XMAX)
				{
					// boundary xmax
					boundary_type = params.boundary_type_xmax;

					j = index / ghostWidth;
					i = index - j*ghostWidth;
					i += (nx+ghostWidth);

					if(j >= ghostWidth && j <= ny+ghostWidth-1    &&
							i >= nx+ghostWidth && i <= nx+2*ghostWidth-1)
					{
						real_t sign=1.0;
						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if (iVar!=IH)
							{
								if ( boundary_type == BC_DIRICHLET )
								{
									i0=2*nx+2*ghostWidth-1-i;
									if (iVar==IU) sign=-ONE_F;
								} else if ( boundary_type == BC_NEUMANN )
								{
									i0=nx+ghostWidth-1;
								}
								else
								{
									// periodic
									i0=i-nx;
								}

								Udata(i  ,j  , iVar) = Udata(i0 ,j  , iVar)*sign;
							}
						}
					}
				} // end FACE_XMAX

				if (faceId == FACE_YMIN)
				{
					// boundary ymin
					boundary_type = params.boundary_type_ymin;

					i = index / ghostWidth;
					j = index - i*ghostWidth;

					if(i >= imin && i <= imax    &&
							j >= 0    && j <ghostWidth)
					{
						real_t sign=1.0;

						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if (iVar!=IH)
							{
								if ( boundary_type == BC_DIRICHLET )
								{
									j0=2*ghostWidth-1-j;
									if (iVar==IV) sign=-ONE_F;
								}
								else if ( boundary_type == BC_NEUMANN )
								{
									j0=ghostWidth;
								}
								else
								{
									// periodic
									j0=ny+j;
								}

								Udata(i  ,j  , iVar) = Udata(i  ,j0 , iVar)*sign;
							}
						}
					}
				} // end FACE_YMIN

				if (faceId == FACE_YMAX)
				{
					// boundary ymax
					boundary_type = params.boundary_type_ymax;

					i = index / ghostWidth;
					j = index - i*ghostWidth;
					j += (ny+ghostWidth);
					if(i >= imin          && i <= imax              &&
							j >= ny+ghostWidth && j <= ny+2*ghostWidth-1)
					{

						real_t sign=1.0;
						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if (iVar!=IH)
							{
								if ( boundary_type == BC_DIRICHLET )
								{
									j0=2*ny+2*ghostWidth-1-j;
									if (iVar==IV) sign=-ONE_F;
								}
								else if ( boundary_type == BC_NEUMANN )
								{
									j0=ny+ghostWidth-1;
								}
								else
								{
									// periodic
									j0=j-ny;
								}

								Udata(i  ,j  , iVar) = Udata(i  ,j0  , iVar)*sign;
							}
						}
					}
				} // end FACE_YMAX
			} // end operator ()

		HydroParams params;
		DataArray2d Udata;
}; // MakeBoundariesFunctor2D

/**
 * Functors to update ghost cells (Hydro 3D).
 *
 */
template <FaceIdType faceId>
class MakeBoundariesFunctor3D
{
	public:
		MakeBoundariesFunctor3D(HydroParams params,
				DataArray3d Udata) :
			params(params), Udata(Udata) {};

		static void apply(HydroParams params, DataArray3d Udata, int nbCells)
		{
			MakeBoundariesFunctor3D<faceId> functor(params, Udata);
			Kokkos::parallel_for(nbCells, functor);
		}

		KOKKOS_INLINE_FUNCTION
			void operator()(const int& index) const
			{
				const int nx = params.nx;
				const int ny = params.ny;
				const int nz = params.nz;

				const int isize = params.isize;
				const int jsize = params.jsize;
				//const int ksize = params.ksize;
				const int ghostWidth = params.ghostWidth;
				const int nbvar = params.nbvar;

				const int imin = params.imin;
				const int imax = params.imax;

				const int jmin = params.jmin;
				const int jmax = params.jmax;

				const int kmin = params.kmin;
				const int kmax = params.kmax;

				int i,j,k;

				int boundary_type;

				int i0, j0, k0;
				int iVar;

				if (faceId == FACE_XMIN)
				{
					// boundary xmin (index = i + j * ghostWidth + k * ghostWidth*jsize)
					k = index / (ghostWidth*jsize);
					j = (index - k*ghostWidth*jsize) / ghostWidth;
					i = index - j*ghostWidth - k*ghostWidth*jsize;

					boundary_type = params.boundary_type_xmin;

					if(k >= kmin && k <= kmax &&
							j >= jmin && j <= jmax &&
							i >= 0    && i <ghostWidth)
					{

						real_t sign=1.0;
						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if ( boundary_type == BC_DIRICHLET )
							{
								i0=2*ghostWidth-1-i;
								if (iVar==IU) sign=-ONE_F;
							}
							else if( boundary_type == BC_NEUMANN )
							{
								i0=ghostWidth;
							}
							else
							{
								// periodic
								i0=nx+i;
							}

							Udata(i,j,k, iVar) = Udata(i0,j,k, iVar)*sign;

						}

					} // end xmin
				}

				if (faceId == FACE_XMAX)
				{

					// boundary xmax (index = i + j *ghostWidth + k * ghostWidth*jsize)
					// same i,j,k as xmin, except translation along x-axis
					k = index / (ghostWidth*jsize);
					j = (index - k*ghostWidth*jsize) / ghostWidth;
					i = index - j*ghostWidth - k*ghostWidth*jsize;

					i += (nx+ghostWidth);

					boundary_type = params.boundary_type_xmax;

					if(k >= kmin          && k <= kmax &&
							j >= jmin          && j <= jmax &&
							i >= nx+ghostWidth && i <= nx+2*ghostWidth-1)
					{
						real_t sign=1.0;
						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if ( boundary_type == BC_DIRICHLET )
							{
								i0=2*nx+2*ghostWidth-1-i;
								if (iVar==IU) sign=-ONE_F;
							}
							else if ( boundary_type == BC_NEUMANN )
							{
								i0=nx+ghostWidth-1;
							}
							else
							{
								// periodic
								i0=i-nx;
							}

							Udata(i,j,k, iVar) = Udata(i0,j,k, iVar)*sign;
						}
					} // end xmax
				}

				if (faceId == FACE_YMIN)
				{
					// boundary ymin (index = i + j*isize + k*isize*ghostWidth)
					k = index / (isize*ghostWidth);
					j = (index - k*isize*ghostWidth) / isize;
					i = index - j*isize - k*isize*ghostWidth;

					boundary_type = params.boundary_type_ymin;

					if(k >= kmin && k <= kmax       &&
							j >= 0    && j <  ghostWidth &&
							i >= imin && i <= imax)
					{
						real_t sign=1.0;

						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if ( boundary_type == BC_DIRICHLET )
							{
								j0=2*ghostWidth-1-j;
								if (iVar==IV) sign=-ONE_F;
							}
							else if ( boundary_type == BC_NEUMANN )
							{
								j0=ghostWidth;
							}
							else
							{
								// periodic
								j0=ny+j;
							}

							Udata(i,j,k, iVar) = Udata(i,j0,k, iVar)*sign;

						}
					} // end ymin
				}

				if (faceId == FACE_YMAX)
				{
					// boundary ymax (index = i + j*isize + k*isize*ghostWidth)
					// same i,j,k as ymin, except translation along y-axis
					k = index / (isize*ghostWidth);
					j = (index - k*isize*ghostWidth) / isize;
					i = index - j*isize - k*isize*ghostWidth;

					j += (ny+ghostWidth);

					boundary_type = params.boundary_type_ymax;

					if(k >= kmin           && k <= kmax              &&
							j >= ny+ghostWidth  && j <= ny+2*ghostWidth-1 &&
							i >= imin           && i <= imax)
					{

						real_t sign=1.0;
						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if ( boundary_type == BC_DIRICHLET )
							{
								j0=2*ny+2*ghostWidth-1-j;
								if (iVar==IV) sign=-ONE_F;
							}
							else if ( boundary_type == BC_NEUMANN )
							{
								j0=ny+ghostWidth-1;
							}
							else
							{
								// periodic
								j0=j-ny;
							}

							Udata(i,j,k, iVar) = Udata(i,j0,k, iVar)*sign;
						}
					} // end ymax
				}

				if (faceId == FACE_ZMIN)
				{
					// boundary zmin (index = i + j*isize + k*isize*jsize)
					k = index / (isize*jsize);
					j = (index - k*isize*jsize) / isize;
					i = index - j*isize - k*isize*jsize;

					boundary_type = params.boundary_type_zmin;

					if(k >= 0    && k <  ghostWidth &&
							j >= jmin && j <= jmax       &&
							i >= imin && i <= imax)
					{
						real_t sign=1.0;

						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if ( boundary_type == BC_DIRICHLET )
							{
								k0=2*ghostWidth-1-k;
								if (iVar==IW) sign=-ONE_F;
							}
							else if ( boundary_type == BC_NEUMANN )
							{
								k0=ghostWidth;
							}
							else
							{
								// periodic
								k0=nz+k;
							}

							Udata(i,j,k, iVar) = Udata(i,j,k0, iVar)*sign;
						}
					} // end zmin
				}

				if (faceId == FACE_ZMAX)
				{
					// boundary zmax (index = i + j*isize + k*isize*jsize)
					// same i,j,k as ymin, except translation along y-axis
					k = index / (isize*jsize);
					j = (index - k*isize*jsize) / isize;
					i = index - j*isize - k*isize*jsize;

					k += (nz+ghostWidth);

					boundary_type = params.boundary_type_zmax;

					if(k >= nz+ghostWidth && k <= nz+2*ghostWidth-1 &&
							j >= jmin          && j <= jmax              &&
							i >= imin          && i <= imax)
					{
						real_t sign=1.0;

						for ( iVar=0; iVar<nbvar; iVar++ )
						{
							if ( boundary_type == BC_DIRICHLET )
							{
								k0=2*nz+2*ghostWidth-1-k;
								if (iVar==IW) sign=-ONE_F;
							}
							else if ( boundary_type == BC_NEUMANN )
							{
								k0=nz+ghostWidth-1;
							}
							else
							{
								// periodic
								k0=k-nz;
							}

							Udata(i,j,k, iVar) = Udata(i,j,k0, iVar)*sign;
						}
					} // end zmax
				}
			} // end operator ()

		HydroParams params;
		DataArray3d Udata;
}; // MakeBoundariesFunctor3D

#endif // BOUNDARIES_FUNCTORS_H_
