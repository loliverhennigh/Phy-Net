/************************************************************************
 * MechSys - Open Library for Mechanical Systems                        *
 * Copyright (C) 2014 Sergio Galindo                                    *
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


#ifndef MECHSYS_EMLBM_DOMAIN_H
#define MECHSYS_EMLBM_DOMAIN_H

// STD
#include <map>
#include <vector>
#include <utility>
#include <set>

// MechSys
#include <mechsys/emlbm2/Lattice.h>

using std::set;
using std::map;
using std::pair;
using std::make_pair;

namespace EMLBM
{

class Domain
{
public:
    //typedefs
    typedef void (*ptDFun_t) (Domain & Dom, void * UserData);

    //Constructors
    Domain (
    iVec3_t               Ndim,   ///< Cell divisions per side
    double                dx,     ///< Space spacing
    double                dt);    ///< Time step

    //Methods
#ifdef USE_HDF5
    void WriteXDMF         (char const * FileKey);  ///< Write the domain data in xdmf file
#endif

    void Initialize     (double dt=0.0);                                                                                              ///< Set the particles to a initial state and asign the possible insteractions
    void Collide        (size_t Np = 1);                                                                                ///< Apply the interaction forces and the collision operator
    void Solve(double Tf, double dtOut, ptDFun_t ptSetup=NULL, ptDFun_t ptReport=NULL,
    char const * FileKey=NULL, bool RenderVideo=true, size_t Nproc=1);                                                                ///< Solve the Domain dynamics

    //Data
    bool                                         Initialized;         ///< System (particles and interactons) initialized ?
    bool                                              PrtVec;         ///< Print Vector data into the xdmf-h5 files
    bool                                            Finished;         ///< Has the simulation finished
    String                                           FileKey;         ///< File Key for output files
    Lattice                                              Lat;         ///< Fluid Lattices
    double                                              Time;         ///< Time of the simulation
    double                                                dt;         ///< Timestep
    void *                                          UserData;         ///< User Data
    size_t                                           idx_out;         ///< The discrete time step
    size_t                                              Step;         ///< The space step to reduce the size of the h5 file for visualization
    size_t                                             Nproc;         ///< Number of cores used for the simulation
};

inline Domain::Domain(iVec3_t Ndim, double Thedx, double Thedt)
{
    Initialized = false;
    Util::Stopwatch stopwatch;
    printf("\n%s--- Initializing LBM Domain --------------------------------------------%s\n",TERM_CLR1,TERM_RST);
    Lat = Lattice(Ndim,Thedx,Thedt);
    Time   = 0.0;
    dt     = Thedt;
    Step   = 1;
    PrtVec = true;
    printf("%s  Num of cells   = %zd%s\n",TERM_CLR2,Lat.Ncells,TERM_RST);
}


#ifdef USE_HDF5

inline void Domain::WriteXDMF(char const * FileKey)
{
    String fn(FileKey);
    fn.append(".h5");
    hid_t     file_id;
    file_id = H5Fcreate(fn.CStr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    size_t  Nx = Lat.Ndim[0]/Step;
    size_t  Ny = Lat.Ndim[1]/Step;
    size_t  Nz = Lat.Ndim[2]/Step;
    // Creating data sets
    float * Sig       = new float[  Nx*Ny*Nz];
    float * Mu        = new float[  Nx*Ny*Nz];
    float * Eps       = new float[  Nx*Ny*Nz];
    float * Rho       = new float[  Nx*Ny*Nz];
    float * Cur       = new float[3*Nx*Ny*Nz];
    float * Bvec      = new float[3*Nx*Ny*Nz];
    float * Evec      = new float[3*Nx*Ny*Nz];
    float * State     = new float[4*Lat.GetCell(iVec3_t(0,0,0))->Nneigh*Nx*Ny*Nz];

    size_t i=0;
    for (size_t m=0;m<Lat.Ndim(2);m+=Step)
    for (size_t l=0;l<Lat.Ndim(1);l+=Step)
    for (size_t n=0;n<Lat.Ndim(0);n+=Step)
    {
        double sig    = 0.0;
        double mu     = 0.0;
        double eps    = 0.0;
        double cha    = 0.0;
        Vec3_t cur    = OrthoSys::O;
        Vec3_t bvec   = OrthoSys::O;
        Vec3_t evec   = OrthoSys::O;

        for (size_t ni=0;ni<Step;ni++)
        for (size_t li=0;li<Step;li++)
        for (size_t mi=0;mi<Step;mi++)
        {
            sig      += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->Sig;
            mu       += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->Mu;
            eps      += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->Eps;
            cha      += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->Rho;
            cur (0)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->J[0];
            cur (1)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->J[1];
            cur (2)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->J[2];
            bvec(0)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->B[0];
            bvec(1)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->B[1];
            bvec(2)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->B[2];
            evec(0)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->E[0];
            evec(1)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->E[1];
            evec(2)  += Lat.GetCell(iVec3_t(n+ni,l+li,m+mi))->E[2];
        }
        sig  /= Step*Step*Step;
        mu   /= Step*Step*Step;
        eps  /= Step*Step*Step;
        cha  /= Step*Step*Step;
        cur  /= Step*Step*Step;
        bvec /= Step*Step*Step;
        evec /= Step*Step*Step;
        Sig [i]      = (float) sig;
        Mu  [i]      = (float) mu;
        Eps [i]      = (float) eps;
        Rho [i]      = (float) cha;
        Cur [3*i  ]  = (float) cur (0);
        Cur [3*i+1]  = (float) cur (1);
        Cur [3*i+2]  = (float) cur (2);
        Bvec[3*i  ]  = (float) bvec(0);
        Bvec[3*i+1]  = (float) bvec(1);
        Bvec[3*i+2]  = (float) bvec(2);
        Evec[3*i  ]  = (float) evec(0);
        Evec[3*i+1]  = (float) evec(1);
        Evec[3*i+2]  = (float) evec(2);
        i++;
    } 

    size_t i_cell=0;
    for (size_t m=0;m<Lat.Ndim(2);m+=1)
    for (size_t l=0;l<Lat.Ndim(1);l+=1)
    for (size_t n=0;n<Lat.Ndim(0);n+=1)
    for (size_t k=0;k<Lat.GetCell(iVec3_t(0,0,0))->Nneigh;k+=1)
    {
        State[i_cell] = Lat.GetCell(iVec3_t(n,l,m))->FE[0][k];
        i_cell++;
        State[i_cell] = Lat.GetCell(iVec3_t(n,l,m))->FE[1][k];
        i_cell++;
        State[i_cell] = Lat.GetCell(iVec3_t(n,l,m))->FB[0][k];
        i_cell++;
        State[i_cell] = Lat.GetCell(iVec3_t(n,l,m))->FB[1][k];
        i_cell++;
    }

    //Write the data
    hsize_t dims[1];
    dims[0] = Nx*Ny*Nz;
    String dsname;
    dsname.Printf("Charge");
    H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,Rho );
    dsname.Printf("Sigma");
    H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,Sig );
    dsname.Printf("Mu");
    H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,Mu  );
    dsname.Printf("Epsilon");
    H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,Eps );
    if (PrtVec)
    {
        dims[0] = 3*Nx*Ny*Nz;
        dsname.Printf("Current");
        H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,Cur     );
        dsname.Printf("MagField");
        H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,Bvec    );
        dsname.Printf("ElecField");
        H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,Evec    );
    }
    dims[0] = 4*Lat.GetCell(iVec3_t(0,0,0))->Nneigh*Nx*Ny*Nz;
    dsname.Printf("State");
    H5LTmake_dataset_float(file_id,dsname.CStr(),1,dims,State );

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

    delete [] Sig     ;
    delete [] Mu      ;
    delete [] Eps     ;
    delete [] Rho     ;
    delete [] Cur     ;
    delete [] Bvec    ;
    delete [] Evec    ;
    delete [] State    ;


    //Closing the file
    H5Fflush(file_id,H5F_SCOPE_GLOBAL);
    H5Fclose(file_id);

	// Writing xmf fil
    std::ostringstream oss;

    oss << "<?xml version=\"1.0\" ?>\n";
    oss << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    oss << "<Xdmf Version=\"2.0\">\n";
    oss << " <Domain>\n";
    oss << "   <Grid Name=\"EMLBM_Mesh\" GridType=\"Uniform\">\n";
    oss << "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\"/>\n";
    oss << "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
    oss << "       <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\"> 0.0 0.0 0.0\n";
    oss << "       </DataItem>\n";
    oss << "       <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\"> " << Step*Lat.dx << " " << Step*Lat.dx  << " " << Step*Lat.dx  << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Geometry>\n";
    oss << "     <Attribute Name=\"Sigma" << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
    oss << "        " << fn.CStr() <<":/Sigma" << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Attribute>\n";
    oss << "     <Attribute Name=\"Mu" << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
    oss << "        " << fn.CStr() <<":/Mu" << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Attribute>\n";
    oss << "     <Attribute Name=\"Epsilon" << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
    oss << "        " << fn.CStr() <<":/Epsilon" << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Attribute>\n";
    oss << "     <Attribute Name=\"Charge" << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
    oss << "        " << fn.CStr() <<":/Charge" << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Attribute>\n";
    if (PrtVec)
    {
    oss << "     <Attribute Name=\"Current" << "\" AttributeType=\"Vector\" Center=\"Node\">\n";
    oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << " 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
    oss << "        " << fn.CStr() <<":/Current" << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Attribute>\n";
    oss << "     <Attribute Name=\"MagField" << "\" AttributeType=\"Vector\" Center=\"Node\">\n";
    oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << " 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
    oss << "        " << fn.CStr() <<":/MagField" << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Attribute>\n";
    oss << "     <Attribute Name=\"ElecField" << "\" AttributeType=\"Vector\" Center=\"Node\">\n";
    oss << "       <DataItem Dimensions=\"" << Nz << " " << Ny << " " << Nx << " 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n";
    oss << "        " << fn.CStr() <<":/ElecField" << "\n";
    oss << "       </DataItem>\n";
    oss << "     </Attribute>\n";
    }
    oss << "   </Grid>\n";
    oss << " </Domain>\n";
    oss << "</Xdmf>\n";
    fn = FileKey;
    fn.append(".xmf");
    std::ofstream of(fn.CStr(), std::ios::out);
    of << oss.str();
    of.close();
}

#endif

void Domain::Collide (size_t Np)
{
#ifdef USE_OMP
    #pragma omp parallel for schedule (static) num_threads(Np)
#endif
    for (size_t i=0;i<Lat.Ncells;i++)
    {
        Cell * c = Lat.Cells[i];
        c->F0[0] = c->F0[0] - 2.0*(c->F0[0] - c->Rho);
        c->F0[1] = c->F0[1] - 2.0*(c->F0[1] - c->Rho);
        for (size_t k=0;k<c->Nneigh;k++)
        {
            for (size_t mu=0;mu<2;mu++)
            {
                c->FEtemp[mu][k] = c->FE[mu][k] - 2.0*(c->FE[mu][k] - c->FEeq(mu,k));
                c->FBtemp[mu][k] = c->FB[mu][k] - 2.0*(c->FB[mu][k] - c->FBeq(mu,k));
            }
        }
        for (size_t k=0;k<c->Nneigh;k++)
        {
            for (size_t mu=0;mu<2;mu++)
            {
                c->FE[mu][k] = c->FEtemp[mu][k];
                c->FB[mu][k] = c->FBtemp[mu][k];
            }
        }
    }   
}

inline void Domain::Solve(double Tf, double dtOut, ptDFun_t ptSetup, ptDFun_t ptReport,
                          char const * TheFileKey, bool RenderVideo, size_t TheNproc)
{

    idx_out     = 0;
    FileKey.Printf("%s",TheFileKey);
    Finished = false;

    // info
    Util::Stopwatch stopwatch;
    printf("\n%s--- Solving ---------------------------------------------------------------------%s\n",TERM_CLR1   , TERM_RST);
    printf("%s  Time step                        =  %g%s\n"       ,TERM_CLR2, dt                                   , TERM_RST);

    Nproc = TheNproc;

    //for (size_t j=0;j<Lat.Size();j++)
    //{
        //for (size_t i=0;i<Lat[j].Ncells;i++)
        //{
            //Lat[j].Cells[i]->Initialize();
            //Lat[j].Cells[i]->CalcProp();
        //}
    //}



    double tout = Time;
    while (Time < Tf)
    {
        if (ptSetup!=NULL) (*ptSetup) ((*this), UserData);
        if (Time >= tout)
        {
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


#ifdef USE_OMP 
        Collide(Nproc);
        Lat.Stream1  (Nproc);
        Lat.Stream2  (Nproc);
        Lat.CalcField(Nproc);
#endif

        Time += dt;
    }
    // last output
    Finished = true;
    if (ptReport!=NULL) (*ptReport) ((*this), UserData);

    printf("%s  Final CPU time       = %s\n",TERM_CLR2, TERM_RST);
}
}


#endif

