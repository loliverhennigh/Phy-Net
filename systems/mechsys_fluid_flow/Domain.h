/************************************************************************
 * MechSys - Open Library for Mechanical Systems                        *
 * Copyright (C) 2016 Sergio Galindo                                    *
 *                                                                      *
 * This program is free software: you can redistribute it and/or modify *
 * it under the terms of the GNU General Public License as published by *
 * the Free Software Foundation, either version 3 of the License, or    *
 * any later version.                                                   *
 *                                                                      *
 * This program is distributed in the hope that it will be useful,      *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the         *
 * GNU General Public License for more details.                         *
 *                                                                      *
 * You should have received a copy of the GNU General Public License    *
 * along with this program. If not, see <http://www.gnu.org/licenses/>  *
 ************************************************************************/


#ifndef MECHSYS_FLBM_DOMAIN_H
#define MECHSYS_FLBM_DOMAIN_H

// Hdf5
#ifdef USE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#ifdef USE_OCL
#include <mechsys/oclaux/cl.hpp>
#endif

// Std lib
#ifdef USE_OMP
#include <omp.h>
#endif

//STD
#include<iostream>

// Mechsys
#include <mechsys/linalg/matvec.h>
#include <mechsys/util/util.h>
#include <mechsys/util/numstreams.h>
#include <mechsys/util/stopwatch.h>
#include <mechsys/util/numstreams.h>

enum LBMethod
{
    D2Q5,     ///< 2D 5 velocities
    D2Q9,     ///< 2D 9 velocities
    D3Q15,    ///< 3D 15 velocities
    D3Q19,    ///< 3D 19 velocities
    //D3Q27     ///< 3D 27 velocities
};

namespace FLBM
{

#ifdef USE_OCL
typedef struct lbm_aux
{
    size_t        Nl;          ///< Numver of Lattices
    size_t        Nneigh;      ///< Number of Neighbors
    size_t        NCPairs;     ///< Number of cell pairs
    size_t        Nx;          ///< Integer vector with the dimensions of the LBM domain
    size_t        Ny;          ///< Integer vector with the dimensions of the LBM domain
    size_t        Nz;          ///< Integer vector with the dimensions of the LBM domain
    size_t        Op[27];      ///< Array with the opposite directions for bounce back calculation
    cl_double3    C[27];       ///< Collection of discrete velocity vectors
    double        EEk[27];     ///< Dyadic product of discrete velocities for LES calculation
    double        W[27];       ///< Collection of discrete weights
    double        Tau[2];      ///< Collection of characteristic collision times
    double        G[2];        ///< Collection of cohesive constants for multiphase simulation
    double        Gs[2];       ///< Collection of cohesive constants for multiphase simulation
    double        Rhoref[2];   ///< Collection of cohesive constants for multiphase simulation
    double        Psi[2];      ///< Collection of cohesive constants for multiphase simulation
    double        Gmix;        ///< Repulsion constant for multicomponent simulation
    double        Cs;          ///< Lattice speed
    double        Sc;          ///< Smagorinsky constant
    
} d_lbm_aux;
#endif

inline size_t Pt2idx(iVec3_t iv, iVec3_t & Dim) // Calculates the index of the cell at coordinates iv for a cubic lattice of dimensions Dim
{
    return iv(0) + iv(1)*Dim(0) + iv(2)*Dim(0)*Dim(1);
}

inline void   idx2Pt(size_t n, iVec3_t & iv, iVec3_t & Dim) // Calculates the coordinates from the index
{
    iv(0) = n%Dim(0);
    iv(1) = (n/Dim(0))%(Dim(1));
    iv(2) = n/(Dim(0)*Dim(1));
}

class Domain
{
public:
	static const double   WEIGHTSD2Q5   [ 5]; ///< Weights for the equilibrium distribution functions (D2Q5)
	static const double   WEIGHTSD2Q9   [ 9]; ///< Weights for the equilibrium distribution functions (D2Q9)
	static const double   WEIGHTSD3Q15  [15]; ///< Weights for the equilibrium distribution functions (D3Q15)
	static const double   WEIGHTSD3Q19  [19]; ///< Weights for the equilibrium distribution functions (D3Q19)
	//static const double   WEIGHTSD3Q27  [27]; ///< Weights for the equilibrium distribution functions (D3Q27)
	static const Vec3_t   LVELOCD2Q5    [ 5]; ///< Local velocities (D2Q5) 
	static const Vec3_t   LVELOCD2Q9    [ 9]; ///< Local velocities (D2Q9) 
	static const Vec3_t   LVELOCD3Q15   [15]; ///< Local velocities (D3Q15)
	static const Vec3_t   LVELOCD3Q19   [19]; ///< Local velocities (D3Q19)
	//static const Vec3_t   LVELOCD3Q27   [27]; ///< Local velocities (D3Q27)
	static const size_t   OPPOSITED2Q5  [ 5]; ///< Opposite directions (D2Q5) 
	static const size_t   OPPOSITED2Q9  [ 9]; ///< Opposite directions (D2Q9) 
	static const size_t   OPPOSITED3Q15 [15]; ///< Opposite directions (D3Q15)
	static const size_t   OPPOSITED3Q19 [19]; ///< Opposite directions (D3Q19)
	//static const size_t   OPPOSITED3Q27 [27]; ///< Opposite directions (D3Q27)
    
    //typedefs
    typedef void (*ptDFun_t) (Domain & Dom, void * UserData);

    //Constructors
    Domain (LBMethod      Method, ///< Type of array, for example D2Q9
    Array<double>         nu,     ///< Viscosity for each fluid
    iVec3_t               Ndim,   ///< Cell divisions per side
    double                dx,     ///< Space spacing
    double                dt);    ///< Time step

    //Special constructor with only one component, the parameters are the same as above
    Domain (LBMethod      Method, ///< Type of array, for example D2Q9
    double                nu,     ///< Viscosity for each fluid
    iVec3_t               Ndim,   ///< Cell divisions per side
    double                dx,     ///< Space spacing
    double                dt);    ///< Time step

    //Methods
    void   ApplyForcesSC();                                                       ///< Apply the molecular forces for the single component case
    void   ApplyForcesMP();                                                       ///< Apply the molecular forces for the multiphase case
    void   ApplyForcesSCMP();                                                     ///< Apply the molecular forces for the both previous cases
    void   CollideSC();                                                           ///< The collide step of LBM for single component simulations
    void   CollideMP();                                                           ///< The collide step of LBM for multi phase simulations
    void   StreamSC();                                                            ///< The stream step of LBM SC
    void   StreamMP();                                                            ///< The stream step of LBM MP
    void   Initialize(size_t k, iVec3_t idx, double Rho, Vec3_t & Vel);           ///< Initialize each cell with a given density and velocity
    double Feq(size_t k, double Rho, Vec3_t & Vel);                               ///< The equilibrium function
    void Solve(double Tf, double dtOut, ptDFun_t ptSetup=NULL, ptDFun_t ptReport=NULL,
    char const * FileKey=NULL, bool RenderVideo=true, size_t Nproc=1);            ///< Solve the Domain dynamics
    
    //Writing Methods
    void WriteXDMF         (char const * FileKey);                                ///< Write the domain data in xdmf file

    //Methods for OpenCL
    #ifdef USE_OCL
    void UpLoadDevice ();                                                         ///< Upload the buffers into the coprocessor device
    void DnLoadDevice ();                                                         ///< Download the buffers into the coprocessor device
    void ApplyForceCL ();                                                         ///< Apply forces in the coprocessor
    void CollideCL    ();                                                         ///< Apply collision operator in the coprocessor
    void StreamCL     ();                                                         ///< Apply streaming operator in the coprocessor
    #endif
    
    
    #ifdef USE_OMP
    omp_lock_t      lck;                      ///< to protect variables in multithreading
    #endif

    //Data
    double ***** F;                           ///< The array containing the individual functions with the order of the lattice, the x,y,z coordinates and the order of the function.
    double ***** Ftemp;                       ///< A similar array to hold provitional data
    bool   ****  IsSolid;                     ///< An array of bools with an identifier to see if the cell is a solid cell
    Vec3_t ****  Vel;                         ///< The fluid velocities
    Vec3_t ****  BForce;                      ///< Body Force for each cell
    double ****  Rho;                         ///< The fluid densities
    double *     Tau;                         ///< The characteristic time of the lattice
    double *     G;                           ///< The attractive constant for multiphase simulations
    double *     Gs;                          ///< The attractive constant for solid phase
    double *     Psi;                         ///< Parameters for the Shan Chen pseudo potential
    double *     Rhoref;                      ///< Parameters for the Shan Chen pseudo potential
    double       Gmix;                        ///< The mixing constant for multicomponent simulations
    size_t const * Op;                        ///< An array containing the indexes of the opposite direction for bounce back conditions
    double const *  W;                        ///< An array with the direction weights
    double *     EEk;                         ///< Diadic product of the velocity vectors
    Vec3_t const *  C;                        ///< The array of lattice velocities
    size_t       Nneigh;                      ///< Number of Neighbors, depends on the scheme
    double       dt;                          ///< Time Step
    double       dx;                          ///< Grid size
    double       Cs;                          ///< Lattice Velocity
    bool         IsFirstTime;                 ///< Bool variable checking if it is the first time function Setup is called
    iVec3_t      Ndim;                        ///< Lattice Dimensions
    size_t       Ncells;                      ///< Number of cells
    size_t       Nproc;                       ///< Number of processors for openmp
    size_t       idx_out;                     ///< The discrete time step for output
    String       FileKey;                     ///< File Key for output files
    void *       UserData;                    ///< User Data
    size_t       Step;                        ///< Lenght of averaging cube to save data
    double       Time;                        ///< Simulation time variable
    size_t       Nl;                          ///< Number of lattices (fluids)
    double       Sc;                          ///< Smagorinsky constant

    //Array for pair calculation
    size_t       NCellPairs;                  ///< Number of cell pairs
    iVec3_t   *  CellPairs;                   ///< Pairs of cells for molecular force calculation

    #ifdef USE_OCL
    cl::CommandQueue CL_Queue;                ///< CL Queue for coprocessor commands 
    cl::Context      CL_Context;              ///< CL Context for coprocessor commands
    cl::Program      CL_Program;              ///< CL program object containing the kernel routines
    cl::Device       CL_Device;               ///< Identity of the accelrating device
    cl::Buffer     * bF;                      ///< Buffer with the distribution functions
    cl::Buffer     * bFtemp;                  ///< Buffer with the distribution functions temporal
    cl::Buffer     * bIsSolid;                ///< Buffer with the solid bool information
    cl::Buffer     * bBForce;                 ///< Buffer with the body forces
    cl::Buffer     * bVel;                    ///< Buffer with the cell velocities
    cl::Buffer     * bRho;                    ///< Buffer with the cell densities
    cl::Buffer     * bCellPair;               ///< Buffer with the pair cell information for force calculation
    cl::Buffer       blbmaux;                 ///< Buffer with the strcuture containing generic lbm information
    size_t           N_Groups;                ///< Number of work gropous that the GPU can allocate
    #endif
};


inline Domain::Domain(LBMethod TheMethod, Array<double> nu, iVec3_t TheNdim, double Thedx, double Thedt)
{
    Util::Stopwatch stopwatch;
    printf("\n%s--- Initializing LBM Domain --------------------------------------------%s\n",TERM_CLR1,TERM_RST);
    if (nu.Size()==0) throw new Fatal("LBM::Domain: Declare at leat one fluid please");
    if (TheNdim(2) >1&&(TheMethod==D2Q9 ||TheMethod==D2Q5 ))  throw new Fatal("LBM::Domain: D2Q9 scheme does not allow for a third dimension, please set Ndim(2)=1 or change to D3Q15");
    if (TheNdim(2)==1&&(TheMethod==D3Q15||TheMethod==D3Q19))  throw new Fatal("LBM::Domain: Ndim(2) is 1. Either change the method to D2Q9 or increase the z-dimension");
   
    if (TheMethod==D2Q5)
    {
        Nneigh = 5;
        W      = WEIGHTSD2Q5;
        C      = LVELOCD2Q5;
        Op     = OPPOSITED2Q5;
    }
    if (TheMethod==D2Q9)
    {
        Nneigh = 9;
        W      = WEIGHTSD2Q9;
        C      = LVELOCD2Q9;
        Op     = OPPOSITED2Q9;
    }
    if (TheMethod==D3Q15)
    {
        Nneigh = 15;
        W      = WEIGHTSD3Q15;
        C      = LVELOCD3Q15;
        Op     = OPPOSITED3Q15;
    }
    if (TheMethod==D3Q19)
    {
        Nneigh = 19;
        W      = WEIGHTSD3Q19;
        C      = LVELOCD3Q19;
        Op     = OPPOSITED3Q19;
    }
    

    Time        = 0.0;
    dt          = Thedt;
    dx          = Thedx;
    Cs          = dx/dt;
    Step        = 1;
    Sc          = 0.17;
    Nl          = nu.Size();
    Ndim        = TheNdim;
    Ncells      = Ndim(0)*Ndim(1)*Ndim(2);
    IsFirstTime = true;


    Tau = new double [Nl];
    G   = new double [Nl];
    Gs  = new double [Nl];
    Rhoref = new double [Nl];
    Psi    = new double [Nl];
    Gmix= 0.0;

    F       = new double **** [Nl];
    Ftemp   = new double **** [Nl];
    Vel     = new Vec3_t ***  [Nl];
    BForce  = new Vec3_t ***  [Nl];
    Rho     = new double ***  [Nl];
    IsSolid = new bool   ***  [Nl];

    for (size_t i=0;i<Nl;i++)
    {
        Tau     [i]    = 3.0*nu[i]*dt/(dx*dx)+0.5;
        G       [i]    = 0.0;
        Gs      [i]    = 0.0;
        Rhoref  [i]    = 200.0;
        Psi     [i]    = 4.0;
        F       [i]    = new double *** [Ndim(0)];
        Ftemp   [i]    = new double *** [Ndim(0)];
        Vel     [i]    = new Vec3_t **  [Ndim(0)];
        BForce  [i]    = new Vec3_t **  [Ndim(0)];
        Rho     [i]    = new double **  [Ndim(0)];
        IsSolid [i]    = new bool   **  [Ndim(0)];
        for (size_t nx=0;nx<Ndim(0);nx++)
        {
            F       [i][nx]    = new double ** [Ndim(1)];
            Ftemp   [i][nx]    = new double ** [Ndim(1)];
            Vel     [i][nx]    = new Vec3_t *  [Ndim(1)];
            BForce  [i][nx]    = new Vec3_t *  [Ndim(1)];
            Rho     [i][nx]    = new double *  [Ndim(1)];
            IsSolid [i][nx]    = new bool   *  [Ndim(1)];
            for (size_t ny=0;ny<Ndim(1);ny++)
            {
                F       [i][nx][ny]    = new double * [Ndim(2)];
                Ftemp   [i][nx][ny]    = new double * [Ndim(2)];
                Vel     [i][nx][ny]    = new Vec3_t   [Ndim(2)];
                BForce  [i][nx][ny]    = new Vec3_t   [Ndim(2)];
                Rho     [i][nx][ny]    = new double   [Ndim(2)];
                IsSolid [i][nx][ny]    = new bool     [Ndim(2)];
                for (size_t nz=0;nz<Ndim(2);nz++)
                {
                    F    [i][nx][ny][nz]    = new double [Nneigh];
                    Ftemp[i][nx][ny][nz]    = new double [Nneigh];
                    IsSolid[i][nx][ny][nz]  = false;
                    for (size_t nn=0;nn<Nneigh;nn++)
                    {
                        F    [i][nx][ny][nz][nn] = 0.0;
                        Ftemp[i][nx][ny][nz][nn] = 0.0;
                    }
                }
            }
        }
    }


    EEk = new double [Nneigh];
    for (size_t k=0;k<Nneigh;k++)
    {
        EEk[k]    = 0.0;
        for (size_t n=0;n<3;n++)
        for (size_t m=0;m<3;m++)
        {
            EEk[k] += fabs(C[k][n]*C[k][m]);
        }
    }

    printf("%s  Num of cells   = %zd%s\n",TERM_CLR2,Nl*Ncells,TERM_RST);
#ifdef USE_OMP
    omp_init_lock(&lck);
#endif
}

inline Domain::Domain(LBMethod TheMethod, double Thenu, iVec3_t TheNdim, double Thedx, double Thedt)
{
    Array<double> nu(1);
    nu[0] = Thenu;
    
    Util::Stopwatch stopwatch;
    printf("\n%s--- Initializing LBM Domain --------------------------------------------%s\n",TERM_CLR1,TERM_RST);
    if (nu.Size()==0) throw new Fatal("LBM::Domain: Declare at leat one fluid please");
    if (TheNdim(2) >1&&(TheMethod==D2Q9 ||TheMethod==D2Q5 ))  throw new Fatal("LBM::Domain: D2Q9 scheme does not allow for a third dimension, please set Ndim(2)=1 or change to D3Q15");
    if (TheNdim(2)==1&&(TheMethod==D3Q15||TheMethod==D3Q19))  throw new Fatal("LBM::Domain: Ndim(2) is 1. Either change the method to D2Q9 or increase the z-dimension");
   
    if (TheMethod==D2Q5)
    {
        Nneigh = 5;
        W      = WEIGHTSD2Q5;
        C      = LVELOCD2Q5;
        Op     = OPPOSITED2Q5;
    }
    if (TheMethod==D2Q9)
    {
        Nneigh = 9;
        W      = WEIGHTSD2Q9;
        C      = LVELOCD2Q9;
        Op     = OPPOSITED2Q9;
    }
    if (TheMethod==D3Q15)
    {
        Nneigh = 15;
        W      = WEIGHTSD3Q15;
        C      = LVELOCD3Q15;
        Op     = OPPOSITED3Q15;
    }
    if (TheMethod==D3Q19)
    {
        Nneigh = 19;
        W      = WEIGHTSD3Q19;
        C      = LVELOCD3Q19;
        Op     = OPPOSITED3Q19;
    }
    

    Time        = 0.0;
    dt          = Thedt;
    dx          = Thedx;
    Cs          = dx/dt;
    Step        = 1;
    Sc          = 0.17;
    Nl          = 1;
    Ndim        = TheNdim;
    Ncells      = Ndim(0)*Ndim(1)*Ndim(2);
    IsFirstTime = true;


    Tau    = new double [Nl];
    G      = new double [Nl];
    Gs     = new double [Nl];
    Rhoref = new double [Nl];
    Psi    = new double [Nl];
    Gmix   = 0.0;

    F       = new double **** [Nl];
    Ftemp   = new double **** [Nl];
    Vel     = new Vec3_t ***  [Nl];
    BForce  = new Vec3_t ***  [Nl];
    Rho     = new double ***  [Nl];
    IsSolid = new bool   ***  [Nl];

    for (size_t i=0;i<Nl;i++)
    {
        Tau     [i]    = 3.0*nu[i]*dt/(dx*dx)+0.5;
        G       [i]    = 0.0;
        Gs      [i]    = 0.0;
        Rhoref  [i]    = 200.0;
        Psi     [i]    = 4.0;
        F       [i]    = new double *** [Ndim(0)];
        Ftemp   [i]    = new double *** [Ndim(0)];
        Vel     [i]    = new Vec3_t **  [Ndim(0)];
        BForce  [i]    = new Vec3_t **  [Ndim(0)];
        Rho     [i]    = new double **  [Ndim(0)];
        IsSolid [i]    = new bool   **  [Ndim(0)];
        for (size_t nx=0;nx<Ndim(0);nx++)
        {
            F       [i][nx]    = new double ** [Ndim(1)];
            Ftemp   [i][nx]    = new double ** [Ndim(1)];
            Vel     [i][nx]    = new Vec3_t *  [Ndim(1)];
            BForce  [i][nx]    = new Vec3_t *  [Ndim(1)];
            Rho     [i][nx]    = new double *  [Ndim(1)];
            IsSolid [i][nx]    = new bool   *  [Ndim(1)];
            for (size_t ny=0;ny<Ndim(1);ny++)
            {
                F       [i][nx][ny]    = new double * [Ndim(2)];
                Ftemp   [i][nx][ny]    = new double * [Ndim(2)];
                Vel     [i][nx][ny]    = new Vec3_t   [Ndim(2)];
                BForce  [i][nx][ny]    = new Vec3_t   [Ndim(2)];
                Rho     [i][nx][ny]    = new double   [Ndim(2)];
                IsSolid [i][nx][ny]    = new bool     [Ndim(2)];
                for (size_t nz=0;nz<Ndim(2);nz++)
                {
                    F    [i][nx][ny][nz]    = new double [Nneigh];
                    Ftemp[i][nx][ny][nz]    = new double [Nneigh];
                    IsSolid[i][nx][ny][nz]  = false;
                    for (size_t nn=0;nn<Nneigh;nn++)
                    {
                        F    [i][nx][ny][nz][nn] = 0.0;
                        Ftemp[i][nx][ny][nz][nn] = 0.0;
                    }
                }
            }
        }
    }


    EEk = new double [Nneigh];
    for (size_t k=0;k<Nneigh;k++)
    {
        EEk[k]    = 0.0;
        for (size_t n=0;n<3;n++)
        for (size_t m=0;m<3;m++)
        {
            EEk[k] += fabs(C[k][n]*C[k][m]);
        }
    }

    printf("%s  Num of cells   = %zd%s\n",TERM_CLR2,Nl*Ncells,TERM_RST);
#ifdef USE_OMP
    omp_init_lock(&lck);
#endif
}

inline void Domain::WriteXDMF(char const * FileKey)
{
    String fn(FileKey);
    fn.append(".h5");
    hid_t     file_id;
    file_id = H5Fcreate(fn.CStr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    size_t  Nx = Ndim[0]/Step;
    size_t  Ny = Ndim[1]/Step;
    size_t  Nz = Ndim[2]/Step;

    for (size_t j=0;j<Nl;j++)
    {
        // Creating data sets
        double * Density   = new double[  Nx*Ny*Nz];
        double * Gamma     = new double[  Nx*Ny*Nz];
        double * Vvec      = new double[3*Nx*Ny*Nz];
        double * Sstate    = new double[Nneigh*Nx*Ny*Nz];

        size_t i=0;
        for (size_t m=0;m<Ndim(2);m+=Step)
        for (size_t l=0;l<Ndim(1);l+=Step)
        for (size_t n=0;n<Ndim(0);n+=Step)
        {
            double rho    = 0.0;
            double gamma  = 0.0;
            Vec3_t vel    = OrthoSys::O;

            for (size_t ni=0;ni<Step;ni++)
            for (size_t li=0;li<Step;li++)
            for (size_t mi=0;mi<Step;mi++)
            {
                rho    += Rho    [j][n+ni][l+li][m+mi];
                gamma  += IsSolid[j][n+ni][l+li][m+mi] ? 1.0: 0.0;
                vel    += Vel    [j][n+ni][l+li][m+mi];
            }
            rho  /= Step*Step*Step;
            gamma/= Step*Step*Step;
            vel  /= Step*Step*Step;
            Density [i]  = (double) rho;
            Gamma   [i]  = (double) gamma;
            Vvec[3*i  ]  = (double) vel(0);
            Vvec[3*i+1]  = (double) vel(1);
            Vvec[3*i+2]  = (double) vel(2);
            i++;
        }
        
        size_t i_state=0;
        for (size_t m=0;m<Ndim(2);m+=1)
        for (size_t l=0;l<Ndim(1);l+=1)
        for (size_t n=0;n<Ndim(0);n+=1)
        {
            for (size_t k=0;k<Nneigh;k+=1)
            {
                Sstate[i_state]  = (double) F[j][n][l][m][k];
                i_state++;
            }
        }



        //Writing data to h5 file
        hsize_t dims[1];
        dims[0] = Nx*Ny*Nz;
        String dsname;
        dsname.Printf("Density_%d",j);
        H5LTmake_dataset_double(file_id,dsname.CStr(),1,dims,Density );
        if (j==0)
        {
            dsname.Printf("Gamma");
            H5LTmake_dataset_double(file_id,dsname.CStr(),1,dims,Gamma   );
        }
        dims[0] = 3*Nx*Ny*Nz;
        dsname.Printf("Velocity_%d",j);
        H5LTmake_dataset_double(file_id,dsname.CStr(),1,dims,Vvec    );
        // write full state of simulation
        dims[0] = Nneigh*Nx*Ny*Nz;
        dsname.Printf("State_%d",j);
        H5LTmake_dataset_double(file_id,dsname.CStr(),1,dims,Sstate    );

        dims[0] = 1;
        int N[1];
        N[0] = Nx;
        dsname.Printf("Nx");
        H5LTmake_dataset_int(file_id,dsname.CStr(),1,dims,N);
        dims[0] = 1;
        N[0] = Ny;
        dsname.Printf("Ny");
        H5LTmake_dataset_int(file_id,dsname.CStr(),1,dims,N);
        dims[0] = 1;
        N[0] = Nz;
        dsname.Printf("Nz");
        H5LTmake_dataset_int(file_id,dsname.CStr(),1,dims,N);

        delete [] Density ;
        delete [] Gamma   ;
        delete [] Vvec    ;
    }

    //Closing the file
    H5Fflush(file_id,H5F_SCOPE_GLOBAL);
    H5Fclose(file_id);

	// Writing xmf fil
    std::ostringstream oss;

    //std::cout << "2" << std::endl;

    if (Ndim(2)==1)
    {
        oss << "<?xml version=\"1.0\" ?>\n";
        oss << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
        oss << "<Xdmf Version=\"2.0\">\n";
        oss << " <Domain>\n";
        oss << "   <Grid Name=\"mesh1\" GridType=\"Uniform\">\n";
        oss << "     <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"" << Ndim(1) << " " << Ndim(0) << "\"/>\n";
        oss << "     <Geometry GeometryType=\"ORIGIN_DXDY\">\n";
        oss << "       <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\"> 0.0 0.0\n";
        oss << "       </DataItem>\n";
        oss << "       <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\"> 1.0 1.0\n";
        oss << "       </DataItem>\n";
        oss << "     </Geometry>\n";
        for (size_t j=0;j<Nl;j++)
        {
        oss << "     <Attribute Name=\"Density_" << j << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        oss << "       <DataItem Dimensions=\"" << Ndim(0) << " " << Ndim(1) << " " << Ndim(2) << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
        oss << "        " << fn.CStr() <<":/Density_" << j << "\n";
        oss << "       </DataItem>\n";
        oss << "     </Attribute>\n";
        oss << "     <Attribute Name=\"Velocity_" << j << "\" AttributeType=\"Vector\" Center=\"Node\">\n";
        oss << "       <DataItem Dimensions=\"" << Ndim(0) << " " << Ndim(1) << " " << Ndim(2) << " 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
        oss << "        " << fn.CStr() <<":/Velocity_" << j << "\n";
        oss << "       </DataItem>\n";
        oss << "     </Attribute>\n";
        }
        oss << "     <Attribute Name=\"Gamma\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        oss << "       <DataItem Dimensions=\"" << Ndim(0) << " " << Ndim(1) << " " << Ndim(2) << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
        oss << "        " << fn.CStr() <<":/Gamma\n";
        oss << "       </DataItem>\n";
        oss << "     </Attribute>\n";
        oss << "   </Grid>\n";
        oss << " </Domain>\n";
        oss << "</Xdmf>\n";
    }
    
    else
    {
        oss << "<?xml version=\"1.0\" ?>\n";
        oss << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
        oss << "<Xdmf Version=\"2.0\">\n";
        oss << " <Domain>\n";
        oss << "   <Grid Name=\"LBM_Mesh\" GridType=\"Uniform\">\n";
        oss << "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\"/>\n";
        oss << "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
        oss << "       <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\"> 0.0 0.0 0.0\n";
        oss << "       </DataItem>\n";
        oss << "       <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\"> " << Step*dx << " " << Step*dx  << " " << Step*dx  << "\n";
        oss << "       </DataItem>\n";
        oss << "     </Geometry>\n";
        for (size_t j=0;j<Nl;j++)
        {
        oss << "     <Attribute Name=\"Density_" << j << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
        oss << "        " << fn.CStr() <<":/Density_" << j << "\n";
        oss << "       </DataItem>\n";
        oss << "     </Attribute>\n";
        oss << "     <Attribute Name=\"Velocity_" << j << "\" AttributeType=\"Vector\" Center=\"Node\">\n";
        oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << " 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
        oss << "        " << fn.CStr() <<":/Velocity_" << j << "\n";
        oss << "       </DataItem>\n";
        oss << "     </Attribute>\n";
        }
        oss << "     <Attribute Name=\"Gamma\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
        oss << "        " << fn.CStr() <<":/Gamma\n";
        oss << "       </DataItem>\n";
        oss << "     </Attribute>\n";
        oss << "   </Grid>\n";
        oss << " </Domain>\n";
        oss << "</Xdmf>\n";
    }
    fn = FileKey;
    fn.append(".xmf");
    std::ofstream of(fn.CStr(), std::ios::out);
    of << oss.str();
    of.close();

}

inline double Domain::Feq(size_t k, double Rho, Vec3_t & V)
{
    double VdotC = dot(V,C[k]);
    double VdotV = dot(V,V);
    return W[k]*Rho*(1.0 + 3.0*VdotC/Cs + 4.5*VdotC*VdotC/(Cs*Cs) - 1.5*VdotV/(Cs*Cs));
}

inline void Domain::Initialize(size_t il, iVec3_t idx, double TheRho, Vec3_t & TheVel)
{
    size_t ix = idx(0);
    size_t iy = idx(1);
    size_t iz = idx(2);

    BForce[il][ix][iy][iz] = OrthoSys::O;

    for (size_t k=0;k<Nneigh;k++)
    {
        F[il][ix][iy][iz][k] = Feq(k,TheRho,TheVel);
    }

    if (!IsSolid[il][ix][iy][iz])
    {
        Vel[il][ix][iy][iz] = TheVel;
        Rho[il][ix][iy][iz] = TheRho;
    }
    else
    {
        Vel[il][ix][iy][iz] = OrthoSys::O;
        Rho[il][ix][iy][iz] = 0.0;
    }
}

inline void Domain::ApplyForcesSC()
{
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t n=0;n<NCellPairs;n++)
    {
        iVec3_t idxc,idxn;
        size_t k = CellPairs[n](2);
        idx2Pt(CellPairs[n](0),idxc,Ndim);
        idx2Pt(CellPairs[n](1),idxn,Ndim);

        size_t ixc = idxc(0);
        size_t iyc = idxc(1);
        size_t izc = idxc(2);
        size_t ixn = idxn(0);
        size_t iyn = idxn(1);
        size_t izn = idxn(2);

        double psic = 0.0;
        double psin = 0.0;

        IsSolid[0][ixc][iyc][izc] ? psic = 0.0 : psic = Psi[0]*exp(-Rhoref[0]/Rho[0][ixc][iyc][izc]);
        IsSolid[0][ixn][iyn][izn] ? psin = 0.0 : psin = Psi[0]*exp(-Rhoref[0]/Rho[0][ixn][iyn][izn]);

        Vec3_t bforce = -G[0]*W[k]*C[k]*psic*psin;

        BForce[0][ixc][iyc][izc] += bforce;
        BForce[0][ixn][iyn][izn] -= bforce;
    }
}

inline void Domain::ApplyForcesMP()
{
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t n=0;n<NCellPairs;n++)
    {
        iVec3_t idxc,idxn;
        size_t k = CellPairs[n](2);
        idx2Pt(CellPairs[n](0),idxc,Ndim);
        idx2Pt(CellPairs[n](1),idxn,Ndim);

        size_t ixc = idxc(0);
        size_t iyc = idxc(1);
        size_t izc = idxc(2);
        size_t ixn = idxn(0);
        size_t iyn = idxn(1);
        size_t izn = idxn(2);

        double psic = 0.0;
        double psin = 0.0;

        double Gt   = Gmix;

        IsSolid[0][ixc][iyc][izc] ? psic = 1.0, Gt = Gs[1] : psic = Rho[0][ixc][iyc][izc];
        IsSolid[1][ixn][iyn][izn] ? psin = 1.0, Gt = Gs[0] : psin = Rho[1][ixn][iyn][izn];

        Vec3_t bforce = -Gt*W[k]*C[k]*psic*psin;

        BForce[0][ixc][iyc][izc] += bforce;
        BForce[1][ixn][iyn][izn] -= bforce;

        Gt          = Gmix;

        IsSolid[1][ixc][iyc][izc] ? psic = 1.0, Gt = Gs[0] : psic = Rho[1][ixc][iyc][izc];
        IsSolid[0][ixn][iyn][izn] ? psin = 1.0, Gt = Gs[1] : psin = Rho[0][ixn][iyn][izn];

        bforce      = -Gt*W[k]*C[k]*psic*psin;

        BForce[1][ixc][iyc][izc] += bforce;
        BForce[0][ixn][iyn][izn] -= bforce;
    }
}

inline void Domain::ApplyForcesSCMP()
{
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t n=0;n<NCellPairs;n++)
    {
        iVec3_t idxc,idxn;
        size_t k = CellPairs[n](2);
        idx2Pt(CellPairs[n](0),idxc,Ndim);
        idx2Pt(CellPairs[n](1),idxn,Ndim);

        size_t ixc = idxc(0);
        size_t iyc = idxc(1);
        size_t izc = idxc(2);
        size_t ixn = idxn(0);
        size_t iyn = idxn(1);
        size_t izn = idxn(2);

        double psic = 0.0;
        double psin = 0.0;

        IsSolid[0][ixc][iyc][izc] ? psic = 0.0 : psic = Psi[0]*exp(-Rhoref[0]/Rho[0][ixc][iyc][izc]);
        IsSolid[0][ixn][iyn][izn] ? psin = 0.0 : psin = Psi[0]*exp(-Rhoref[0]/Rho[0][ixn][iyn][izn]);

        Vec3_t bforce = -G[0]*W[k]*C[k]*psic*psin;

        BForce[0][ixc][iyc][izc] += bforce;
        BForce[0][ixn][iyn][izn] -= bforce;

        IsSolid[1][ixc][iyc][izc] ? psic = 0.0 : psic = Psi[1]*exp(-Rhoref[1]/Rho[1][ixc][iyc][izc]);
        IsSolid[1][ixn][iyn][izn] ? psin = 0.0 : psin = Psi[1]*exp(-Rhoref[1]/Rho[1][ixn][iyn][izn]);

        bforce        = -G[1]*W[k]*C[k]*psic*psin;

        BForce[1][ixc][iyc][izc] += bforce;
        BForce[1][ixn][iyn][izn] -= bforce;

        double Gt   = Gmix;

        IsSolid[0][ixc][iyc][izc] ? psic = 1.0, Gt = Gs[1] : psic = Rho[0][ixc][iyc][izc];
        IsSolid[1][ixn][iyn][izn] ? psin = 1.0, Gt = Gs[0] : psin = Rho[1][ixn][iyn][izn];

        bforce      = -Gt*W[k]*C[k]*psic*psin;

        BForce[0][ixc][iyc][izc] += bforce;
        BForce[1][ixn][iyn][izn] -= bforce;

        Gt          = Gmix;

        IsSolid[1][ixc][iyc][izc] ? psic = 1.0, Gt = Gs[0] : psic = Rho[1][ixc][iyc][izc];
        IsSolid[0][ixn][iyn][izn] ? psin = 1.0, Gt = Gs[1] : psin = Rho[0][ixn][iyn][izn];

        bforce      = -Gt*W[k]*C[k]*psic*psin;

        BForce[1][ixc][iyc][izc] += bforce;
        BForce[0][ixn][iyn][izn] -= bforce;
    }
}

inline void Domain::CollideSC()
{
    size_t nx = Ndim(0);
    size_t ny = Ndim(1);
    size_t nz = Ndim(2);

    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t ix=0;ix<nx;ix++)
    for (size_t iy=0;iy<ny;iy++)
    for (size_t iz=0;iz<nz;iz++)
    {
        if (!IsSolid[0][ix][iy][iz])
        {
            double NonEq[Nneigh];
            double Q = 0.0;
            double tau = Tau[0];
            double rho = Rho[0][ix][iy][iz];
            Vec3_t vel = Vel[0][ix][iy][iz]+dt*BForce[0][ix][iy][iz]/rho;
            double VdotV = dot(vel,vel);
            for (size_t k=0;k<Nneigh;k++)
            {
                double VdotC = dot(vel,C[k]);
                double Feq   = W[k]*rho*(1.0 + 3.0*VdotC/Cs + 4.5*VdotC*VdotC/(Cs*Cs) - 1.5*VdotV/(Cs*Cs));
                NonEq[k] = F[0][ix][iy][iz][k] - Feq;
                Q +=  NonEq[k]*NonEq[k]*EEk[k];
            }
            Q = sqrt(2.0*Q);
            tau = 0.5*(tau+sqrt(tau*tau + 6.0*Q*Sc/rho));

            bool valid = true;
            double alpha = 1.0;
            while (valid)
            {
                valid = false;
                for (size_t k=0;k<Nneigh;k++)
                {
                    Ftemp[0][ix][iy][iz][k] = F[0][ix][iy][iz][k] - alpha*NonEq[k]/tau;
                    if (Ftemp[0][ix][iy][iz][k]<-1.0e-12)
                    {
                        //std::cout << Ftemp[0][ix][iy][iz][k] << std::endl;
                        double temp =  tau*F[0][ix][iy][iz][k]/NonEq[k];
                        if (temp<alpha) alpha = temp;
                        valid = true;
                    }
                    if (std::isnan(Ftemp[0][ix][iy][iz][k]))
                    {
                        std::cout << "CollideSC: Nan found, resetting" << std::endl;
                        std::cout << " " << alpha << " " << iVec3_t(ix,iy,iz) << " " << k << " " << std::endl;
                        throw new Fatal("Domain::CollideSC: Distribution funcitons gave nan value, check parameters");
                    }
                }
            }
        }
        else
        {
            for (size_t k=0;k<Nneigh;k++)
            {
                Ftemp[0][ix][iy][iz][k] = F[0][ix][iy][iz][Op[k]];
            }
        }
    }

    double ***** tmp = F;
    F = Ftemp;
    Ftemp = tmp;
}

inline void Domain::CollideMP ()
{
    size_t nx = Ndim(0);
    size_t ny = Ndim(1);
    size_t nz = Ndim(2);

    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t ix=0;ix<nx;ix++)
    for (size_t iy=0;iy<ny;iy++)
    for (size_t iz=0;iz<nz;iz++)
    {
        Vec3_t Vmix = OrthoSys::O;
        double den  = 0.0;
        for (size_t il=0;il<Nl;il++)
        {
            Vmix += Rho[il][ix][iy][iz]*Vel[il][ix][iy][iz]/Tau[il];
            den  += Rho[il][ix][iy][iz]/Tau[il];
        }
        Vmix /= den;

        for (size_t il=0;il<Nl;il++)
        {
            if (!IsSolid[il][ix][iy][iz])
            {
                double rho = Rho[il][ix][iy][iz];
                Vec3_t vel = Vmix + Tau[il]*BForce[il][ix][iy][iz]/rho;
                double VdotV = dot(vel,vel);
                bool valid = true;
                double alpha = 1.0;
                while (valid)
                {
                    valid = false;
                    for (size_t k=0;k<Nneigh;k++)
                    {
                        double VdotC = dot(vel,C[k]);
                        double Feq   = W[k]*rho*(1.0 + 3.0*VdotC/Cs + 4.5*VdotC*VdotC/(Cs*Cs) - 1.5*VdotV/(Cs*Cs));
                        Ftemp[il][ix][iy][iz][k] = F[il][ix][iy][iz][k] - alpha*(F[il][ix][iy][iz][k] - Feq)/Tau[il];
                        if (Ftemp[il][ix][iy][iz][k]<-1.0e-12)
                        {
                            //std::cout << Ftemp[0][ix][iy][iz][k] << std::endl;
                            double temp =  Tau[il]*F[il][ix][iy][iz][k]/(F[il][ix][iy][iz][k] - Feq);
                            if (temp<alpha) alpha = temp;
                            valid = true;
                        }
                        if (std::isnan(Ftemp[il][ix][iy][iz][k]))
                        {
                            std::cout << "CollideMP: Nan found, resetting" << std::endl;
                            std::cout << " " << alpha << " " << iVec3_t(ix,iy,iz) << " " << k << " " << std::endl;
                            throw new Fatal("Domain::CollideMP: Distribution funcitons gave nan value, check parameters");
                        }
                    }
                }
            }
            else
            {
                for (size_t k=0;k<Nneigh;k++)
                {
                    Ftemp[il][ix][iy][iz][k] = F[il][ix][iy][iz][Op[k]];
                }
            }
        }
    }

    double ***** tmp = F;
    F = Ftemp;
    Ftemp = tmp;
}

inline void Domain::StreamSC()
{
    size_t nx = Ndim(0);
    size_t ny = Ndim(1);
    size_t nz = Ndim(2);

    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t ix=0;ix<nx;ix++)
    for (size_t iy=0;iy<ny;iy++)
    for (size_t iz=0;iz<nz;iz++)
    {
        for (size_t k=0;k<Nneigh;k++)
        {
            size_t nix = (size_t)((int)ix + (int)C[k](0) + (int)Ndim(0))%Ndim(0);
            size_t niy = (size_t)((int)iy + (int)C[k](1) + (int)Ndim(1))%Ndim(1);
            size_t niz = (size_t)((int)iz + (int)C[k](2) + (int)Ndim(2))%Ndim(2);
            Ftemp[0][nix][niy][niz][k] = F[0][ix][iy][iz][k];
        }
    }

    double ***** tmp = F;
    F = Ftemp;
    Ftemp = tmp;

    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t ix=0;ix<nx;ix++)
    for (size_t iy=0;iy<ny;iy++)
    for (size_t iz=0;iz<nz;iz++)
    {
        BForce[0][ix][iy][iz] = OrthoSys::O;
        Vel   [0][ix][iy][iz] = OrthoSys::O;
        Rho   [0][ix][iy][iz] = 0.0;
        if (!IsSolid[0][ix][iy][iz])
        {
            for (size_t k=0;k<Nneigh;k++)
            {
                Rho[0][ix][iy][iz] +=  F[0][ix][iy][iz][k];
                Vel[0][ix][iy][iz] +=  F[0][ix][iy][iz][k]*C[k];
            }
            Vel[0][ix][iy][iz] /= Rho[0][ix][iy][iz];
        }
    }
}

inline void Domain::StreamMP()
{
    size_t nx = Ndim(0);
    size_t ny = Ndim(1);
    size_t nz = Ndim(2);

    for (size_t il=0;il<Nl;il++)
    {
        #ifdef USE_OMP
        #pragma omp parallel for schedule(static) num_threads(Nproc)
        #endif
        for (size_t ix=0;ix<nx;ix++)
        for (size_t iy=0;iy<ny;iy++)
        for (size_t iz=0;iz<nz;iz++)
        {
            for (size_t k=0;k<Nneigh;k++)
            {
                size_t nix = (size_t)((int)ix + (int)C[k](0) + (int)Ndim(0))%Ndim(0);
                size_t niy = (size_t)((int)iy + (int)C[k](1) + (int)Ndim(1))%Ndim(1);
                size_t niz = (size_t)((int)iz + (int)C[k](2) + (int)Ndim(2))%Ndim(2);
                Ftemp[il][nix][niy][niz][k] = F[il][ix][iy][iz][k];
            }
        }
    }

    double ***** tmp = F;
    F = Ftemp;
    Ftemp = tmp;

    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) num_threads(Nproc)
    #endif
    for (size_t ix=0;ix<nx;ix++)
    for (size_t iy=0;iy<ny;iy++)
    for (size_t iz=0;iz<nz;iz++)
    {
        for (size_t il=0;il<Nl;il++)
        {
            BForce[il][ix][iy][iz] = OrthoSys::O;
            Vel   [il][ix][iy][iz] = OrthoSys::O;
            Rho   [il][ix][iy][iz] = 0.0;
            if (!IsSolid[il][ix][iy][iz])
            {
                for (size_t k=0;k<Nneigh;k++)
                {
                    Rho[il][ix][iy][iz] +=  F[il][ix][iy][iz][k];
                    Vel[il][ix][iy][iz] +=  F[il][ix][iy][iz][k]*C[k];
                }
                Vel[il][ix][iy][iz] /= Rho[il][ix][iy][iz];
            }
        }
    }
}

#ifdef USE_OCL
inline void Domain::UpLoadDevice()
{

    bF         = new cl::Buffer [Nl];
    bFtemp     = new cl::Buffer [Nl];
    bIsSolid   = new cl::Buffer [Nl];
    bBForce    = new cl::Buffer [Nl];
    bVel       = new cl::Buffer [Nl];
    bRho       = new cl::Buffer [Nl];
    
    blbmaux    = cl::Buffer(CL_Context,CL_MEM_READ_WRITE,sizeof(lbm_aux   )              );

    lbm_aux lbmaux[1];

    lbmaux[0].Nx = Ndim(0);
    lbmaux[0].Ny = Ndim(1);
    lbmaux[0].Nz = Ndim(2);
    lbmaux[0].Nneigh    = Nneigh;
    lbmaux[0].NCPairs   = NCellPairs;
    lbmaux[0].Nl        = Nl;
    lbmaux[0].Gmix      = Gmix;
    lbmaux[0].Cs        = Cs;
    lbmaux[0].Sc        = Sc;

    for (size_t nn=0;nn<Nneigh;nn++)
    {
       lbmaux[0].C  [nn].s[0] = C  [nn](0); 
       lbmaux[0].C  [nn].s[1] = C  [nn](1); 
       lbmaux[0].C  [nn].s[2] = C  [nn](2); 
       lbmaux[0].EEk[nn]      = EEk[nn]   ; 
       lbmaux[0].W  [nn]      = W  [nn]   ;
       lbmaux[0].Op [nn]      = Op [nn]   ;
    }

    for (size_t il=0;il<Nl;il++)
    {
        lbmaux[0].Tau     [il]    = Tau     [il];
        lbmaux[0].G       [il]    = G       [il];
        lbmaux[0].Gs      [il]    = Gs      [il];
        lbmaux[0].Rhoref  [il]    = Rhoref  [il];
        lbmaux[0].Psi     [il]    = Psi     [il];

        bF       [il]  = cl::Buffer(CL_Context,CL_MEM_READ_WRITE,sizeof(double    )*Ncells*Nneigh);            
        bFtemp   [il]  = cl::Buffer(CL_Context,CL_MEM_READ_WRITE,sizeof(double    )*Ncells*Nneigh); 
        bIsSolid [il]  = cl::Buffer(CL_Context,CL_MEM_READ_WRITE,sizeof(bool      )*Ncells       );
        bBForce  [il]  = cl::Buffer(CL_Context,CL_MEM_READ_WRITE,sizeof(cl_double3)*Ncells       );     
        bVel     [il]  = cl::Buffer(CL_Context,CL_MEM_READ_WRITE,sizeof(cl_double3)*Ncells       ); 
        bRho     [il]  = cl::Buffer(CL_Context,CL_MEM_READ_WRITE,sizeof(double    )*Ncells       ); 


        bool       *IsSolidCL;
        double     *FCL,*FtempCL,*RhoCL;
        cl_double3 *VelCL,*BForceCL;
        FCL        = new double    [Ncells*Nneigh];
        FtempCL    = new double    [Ncells*Nneigh];
        IsSolidCL  = new bool      [Ncells       ];
        RhoCL      = new double    [Ncells       ];
        VelCL      = new cl_double3[Ncells       ];
        BForceCL   = new cl_double3[Ncells       ];
    
        size_t Nn = 0;
        size_t Nm = 0;
        for (size_t nz=0;nz<Ndim(2);nz++)
        {
            for (size_t ny=0;ny<Ndim(1);ny++)
            {
                for (size_t nx=0;nx<Ndim(0);nx++)
                {
                    IsSolidCL[Nm]      = IsSolid  [il][nx][ny][nz];
                    RhoCL    [Nm]      = Rho      [il][nx][ny][nz];
                    VelCL    [Nm].s[0] = Vel      [il][nx][ny][nz][0];
                    VelCL    [Nm].s[1] = Vel      [il][nx][ny][nz][1];
                    VelCL    [Nm].s[2] = Vel      [il][nx][ny][nz][2];
                    BForceCL [Nm].s[0] = BForce   [il][nx][ny][nz][0];
                    BForceCL [Nm].s[1] = BForce   [il][nx][ny][nz][1];
                    BForceCL [Nm].s[2] = BForce   [il][nx][ny][nz][2];
                    Nm++;
                    for (size_t nn=0;nn<Nneigh;nn++)
                    {
                        FCL    [Nn] = F    [il][nx][ny][nz][nn];
                        FtempCL[Nn] = Ftemp[il][nx][ny][nz][nn];
                        Nn++;
                    }
                }
            }
        }

        CL_Queue.enqueueWriteBuffer(bF      [il],CL_TRUE,0,sizeof(double    )*Ncells*Nneigh,FCL       );  
        CL_Queue.enqueueWriteBuffer(bFtemp  [il],CL_TRUE,0,sizeof(double    )*Ncells*Nneigh,FtempCL   );  
        CL_Queue.enqueueWriteBuffer(bIsSolid[il],CL_TRUE,0,sizeof(bool      )*Ncells       ,IsSolidCL );  
        CL_Queue.enqueueWriteBuffer(bBForce [il],CL_TRUE,0,sizeof(cl_double3)*Ncells       ,BForceCL  );  
        CL_Queue.enqueueWriteBuffer(bVel    [il],CL_TRUE,0,sizeof(cl_double3)*Ncells       ,VelCL     );  
        CL_Queue.enqueueWriteBuffer(bRho    [il],CL_TRUE,0,sizeof(double    )*Ncells       ,RhoCL     );  
        

        delete [] FCL       ;
        delete [] FtempCL   ;
        delete [] IsSolidCL ;
        delete [] RhoCL     ;
        delete [] VelCL     ;
        delete [] BForceCL  ;
    }
    CL_Queue.enqueueWriteBuffer(blbmaux  ,CL_TRUE,0,sizeof(lbm_aux   )              ,lbmaux    );  

    cl::Kernel kernel = cl::Kernel(CL_Program,"CheckUpLoad");
    kernel.setArg(0,blbmaux );
    CL_Queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(1),cl::NullRange);
    CL_Queue.finish();
}

inline void Domain::DnLoadDevice()
{
    for (size_t il=0;il<Nl;il++)
    {
        double     * RhoCL;
        cl_double3 * VelCL;
        double     * FCL;
        RhoCL      = new double    [Ncells];
        VelCL      = new cl_double3[Ncells];
        FCL        = new double    [Nneigh*Ncells];

        CL_Queue.enqueueReadBuffer(bRho[il],CL_TRUE,0,sizeof(double    )*Ncells,RhoCL);
        CL_Queue.enqueueReadBuffer(bVel[il],CL_TRUE,0,sizeof(cl_double3)*Ncells,VelCL);
        CL_Queue.enqueueReadBuffer(bF[il],CL_TRUE,0,sizeof(double      )*Ncells*Nneigh,FCL);

        size_t Nm = 0;
        size_t Nfm = 0;
        for (size_t nz=0;nz<Ndim(2);nz++)
        {
            for (size_t ny=0;ny<Ndim(1);ny++)
            {
                for (size_t nx=0;nx<Ndim(0);nx++)
                {
                    Rho[il][nx][ny][nz]    =  RhoCL[Nm]     ;
                    Vel[il][nx][ny][nz][0] =  VelCL[Nm].s[0];
                    Vel[il][nx][ny][nz][1] =  VelCL[Nm].s[1];
                    Vel[il][nx][ny][nz][2] =  VelCL[Nm].s[2];
                    for (size_t nn=0;nn<Nneigh;nn++)
                    {
                        F[il][nx][ny][nz][nn] =  FCL[Nfm];
                        Nfm++;
                    }
                    Nm++;
                }
            }
        }
        delete [] RhoCL     ;
        delete [] VelCL     ;
        delete [] FCL     ;
    }
}

inline void Domain::ApplyForceCL()
{
    cl::Kernel kernel;
    if (Nl==1)
    {
        kernel = cl::Kernel(CL_Program,"ApplyForcesSC"  );
        kernel.setArg(0,bIsSolid[0]);
        kernel.setArg(1,bBForce [0]);
        kernel.setArg(2,bRho    [0]);
        kernel.setArg(3,blbmaux    );
        CL_Queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(Ncells),cl::NullRange);
        CL_Queue.finish();
    }
    else if ((fabs(G[0])>1.0e-12)||(fabs(G[1])>1.0e-12))
    {
        kernel = cl::Kernel(CL_Program,"ApplyForcesSCMP");
        kernel.setArg(0,bIsSolid[0]);
        kernel.setArg(1,bIsSolid[1]);
        kernel.setArg(2,bBForce [0]);
        kernel.setArg(3,bBForce [1]);
        kernel.setArg(4,bRho    [0]);
        kernel.setArg(5,bRho    [1]);
        kernel.setArg(6,blbmaux    );
        CL_Queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(Ncells),cl::NullRange);
        CL_Queue.finish();
    }
    else
    {
        kernel = cl::Kernel(CL_Program,"ApplyForcesMP"  );
        kernel.setArg(0,bIsSolid[0]);
        kernel.setArg(1,bIsSolid[1]);
        kernel.setArg(2,bBForce [0]);
        kernel.setArg(3,bBForce [1]);
        kernel.setArg(4,bRho    [0]);
        kernel.setArg(5,bRho    [1]);
        kernel.setArg(6,blbmaux    );
        CL_Queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(Ncells),cl::NullRange);
        CL_Queue.finish();
    }
}

inline void Domain::CollideCL()
{
    cl::Kernel kernel;
    if (Nl==1)   
    {
        kernel = cl::Kernel(CL_Program,"CollideSC");
        kernel.setArg(0,bIsSolid[0]);
        kernel.setArg(1,bF      [0]);
        kernel.setArg(2,bFtemp  [0]);
        kernel.setArg(3,bBForce [0]);
        kernel.setArg(4,bVel    [0]);
        kernel.setArg(5,bRho    [0]);
        kernel.setArg(6,blbmaux    );
    }
    else
    {        
        kernel = cl::Kernel(CL_Program,"CollideMP");
        kernel.setArg( 0,bIsSolid[0]);
        kernel.setArg( 1,bIsSolid[1]);
        kernel.setArg( 2,bF      [0]);
        kernel.setArg( 3,bF      [1]);
        kernel.setArg( 4,bFtemp  [0]);
        kernel.setArg( 5,bFtemp  [1]);
        kernel.setArg( 6,bBForce [0]);
        kernel.setArg( 7,bBForce [1]);
        kernel.setArg( 8,bVel    [0]);
        kernel.setArg( 9,bVel    [1]);
        kernel.setArg(10,bRho    [0]);
        kernel.setArg(11,bRho    [1]);
        kernel.setArg(12,blbmaux    );
    }
    CL_Queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(Ncells),cl::NullRange);
    CL_Queue.finish();
}

inline void Domain::StreamCL()
{
    for (size_t il=0;il<Nl;il++)
    {
        cl::Kernel kernel = cl::Kernel(CL_Program,"Stream1");
        kernel.setArg(0,bF      [il]);
        kernel.setArg(1,bFtemp  [il]);
        kernel.setArg(2,blbmaux     );
        CL_Queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(Ncells),cl::NullRange);
        CL_Queue.finish();

        kernel            = cl::Kernel(CL_Program,"Stream2");
        kernel.setArg(0,bIsSolid[il]);
        kernel.setArg(1,bF      [il]);
        kernel.setArg(2,bFtemp  [il]);
        kernel.setArg(3,bBForce [il]);
        kernel.setArg(4,bVel    [il]);
        kernel.setArg(5,bRho    [il]);
        kernel.setArg(6,blbmaux     );
        CL_Queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(Ncells),cl::NullRange);
        CL_Queue.finish();
    }
}
#endif


inline void Domain::Solve(double Tf, double dtOut, ptDFun_t ptSetup, ptDFun_t ptReport,
                          char const * TheFileKey, bool RenderVideo, size_t TheNproc)
{
    #ifdef USE_OCL 
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size()==0)
    {
        throw new Fatal("FLBM::Domain: There are no GPUs or APUs, please compile with the A_USE_OCL flag turned off");
    }
    cl::Platform default_platform=all_platforms[0];
    std::vector<cl::Device> all_devices; 
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size()==0)
    {
        throw new Fatal("FLBM::Domain: There are no GPUs or APUs, please compile with the A_USE_OCL flag turned off");
    }
    CL_Device  = all_devices[0];
    CL_Context = cl::Context(CL_Device);
 
    cl::Program::Sources sources;

    char* pMECHSYS_ROOT;
    pMECHSYS_ROOT = getenv ("MECHSYS_ROOT");
    if (pMECHSYS_ROOT==NULL) pMECHSYS_ROOT = getenv ("HOME");

    String pCL;
    pCL.Printf("%s/mechsys/lib/flbm/lbm.cl",pMECHSYS_ROOT);

    std::ifstream infile(pCL.CStr(),std::ifstream::in);
    std::string kernel_code((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());

    sources.push_back({kernel_code.c_str(),kernel_code.length()}); 

    CL_Program = cl::Program(CL_Context,sources);
    if(CL_Program.build({CL_Device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<CL_Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(CL_Device)<<"\n";
        exit(1);
    }

    CL_Queue   = cl::CommandQueue(CL_Context,CL_Device);

    N_Groups   = CL_Device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    #endif


    idx_out     = 0;
    FileKey.Printf("%s",TheFileKey);
    Nproc = TheNproc;

    Util::Stopwatch stopwatch;
    printf("\n%s--- Solving ---------------------------------------------------------------------%s\n",TERM_CLR1    , TERM_RST);
    printf("%s  Time step                        =  %g%s\n"       ,TERM_CLR2, dt                                    , TERM_RST);
    for (size_t i=0;i<Nl;i++)
    {
    printf("%s  Tau of Lattice %zd                 =  %g%s\n"       ,TERM_CLR2, i, Tau[i]                             , TERM_RST);
    }
    #ifdef USE_OCL
    //printf("%s  Using GPU:                       =  %c%s\n"     ,TERM_CLR2, Default_Device.getInfo<CL_DEVICE_NAME>(), TERM_RST);
    std::cout 
        << TERM_CLR2 
        << "  Using GPU:                       =  " << CL_Device.getInfo<CL_DEVICE_NAME>() << TERM_RST << std::endl;
    #endif
   
    {
        Array<iVec3_t> CPP(0);

        size_t nx = Ndim(0);
        size_t ny = Ndim(1);
        size_t nz = Ndim(2);

        for (size_t iz=0;iz<nz;iz++)
        for (size_t iy=0;iy<ny;iy++)
        for (size_t ix=0;ix<nx;ix++)
        {
            size_t nc = Pt2idx(iVec3_t(ix,iy,iz),Ndim);
            for (size_t k=1;k<Nneigh;k++)
            {
                size_t nix = (size_t)((int)ix + (int)C[k](0) + (int)Ndim(0))%Ndim(0);
                size_t niy = (size_t)((int)iy + (int)C[k](1) + (int)Ndim(1))%Ndim(1);
                size_t niz = (size_t)((int)iz + (int)C[k](2) + (int)Ndim(2))%Ndim(2);
                size_t nb  = Pt2idx(iVec3_t(nix,niy,niz),Ndim);
                if (nb>nc)
                {
                    CPP.Push(iVec3_t(nc,nb,k));
                }
            }
        }

        NCellPairs = CPP.Size();
        CellPairs = new iVec3_t [NCellPairs];
        for (size_t n=0;n<NCellPairs;n++)
        {
            CellPairs[n] = CPP[n];
        }
    }

    //std::cout << "1" << std::endl;
    #ifdef USE_OCL
    UpLoadDevice();
    #endif
    
    //std::cout << "2" << std::endl;

    double tout = Time;
    while (Time < Tf)
    {
        if (ptSetup!=NULL) (*ptSetup) ((*this), UserData);
        if (Time >= tout)
        {
            //std::cout << "3" << std::endl;
            #ifdef USE_OCL
            DnLoadDevice();
            #endif
            //std::cout << "4" << std::endl;
            if (TheFileKey!=NULL)
            {
                String fn;
                fn.Printf    ("%s_%04d", TheFileKey, idx_out);
                if ( RenderVideo) 
                {
                    #ifdef USE_HDF5
                    WriteXDMF(fn.CStr());
                    #else
                    //WriteVTK (fn.CStr());
                    #endif
                }
                if (ptReport!=NULL) (*ptReport) ((*this), UserData);
            }
            tout += dtOut;
            idx_out++;
        }

        //The LBM dynamics
        #ifdef USE_OCL
        ApplyForceCL();
        CollideCL();
        StreamCL();
        #else
        if (Nl==1)
        {
            if (fabs(G[0])>1.0e-12) ApplyForcesSC();
            CollideSC();
            StreamSC();
        }
        else
        {
            if ((fabs(G[0])>1.0e-12)||(fabs(G[1])>1.0e-12)) ApplyForcesSCMP();
            else ApplyForcesMP();
            CollideMP();
            StreamMP();
        }
        #endif

        Time += dt;
        //std::cout << Time << std::endl;
    }
}

const double Domain::WEIGHTSD2Q5   [ 5] = { 2./6., 1./6., 1./6., 1./6., 1./6 };
const double Domain::WEIGHTSD2Q9   [ 9] = { 4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36. };
const double Domain::WEIGHTSD3Q15  [15] = { 2./9., 1./9., 1./9., 1./9., 1./9.,  1./9.,  1./9., 1./72., 1./72. , 1./72., 1./72., 1./72., 1./72., 1./72., 1./72.};
const double Domain::WEIGHTSD3Q19  [19] = { 1./3., 1./18., 1./18., 1./18., 1./18., 1./18., 1./18., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.};
const size_t Domain::OPPOSITED2Q5  [ 5] = { 0, 3, 4, 1, 2 };                                                       ///< Opposite directions (D2Q5) 
const size_t Domain::OPPOSITED2Q9  [ 9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };                                           ///< Opposite directions (D2Q9) 
const size_t Domain::OPPOSITED3Q15 [15] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13};                     ///< Opposite directions (D3Q15)
const size_t Domain::OPPOSITED3Q19 [19] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};     ///< Opposite directions (D3Q19)
const Vec3_t Domain::LVELOCD2Q5  [ 5] = { {0,0,0}, {1,0,0}, {0,1,0}, {-1,0,0}, {0,-1,0} };
const Vec3_t Domain::LVELOCD2Q9  [ 9] = { {0,0,0}, {1,0,0}, {0,1,0}, {-1,0,0}, {0,-1,0}, {1,1,0}, {-1,1,0}, {-1,-1,0}, {1,-1,0} };
const Vec3_t Domain::LVELOCD3Q15 [15] =
{
	{ 0, 0, 0}, { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, 
	{ 0, 0, 1}, { 0, 0,-1}, { 1, 1, 1}, {-1,-1,-1}, { 1, 1,-1}, 
	{-1,-1, 1}, { 1,-1, 1}, {-1, 1,-1}, { 1,-1,-1}, {-1, 1, 1} 
};
const Vec3_t Domain::LVELOCD3Q19 [19] =
{
	{ 0, 0, 0}, 
    { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1}, 
    { 1, 1, 0}, {-1,-1, 0}, { 1,-1, 0}, {-1, 1, 0}, { 1, 0, 1}, {-1, 0,-1},
    { 1, 0,-1}, {-1, 0, 1}, { 0, 1, 1}, { 0,-1,-1}, { 0, 1,-1}, { 0,-1, 1}
};

}
#endif
