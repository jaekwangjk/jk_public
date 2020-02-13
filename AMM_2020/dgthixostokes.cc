#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include "headers/post_processor.h"
#include "headers/precondition.h"
#include "headers/simple_functions.h"
#include "headers/bc_and_rhs.h"

// Maximum number of refinement cycle (adaptive mesh refinement)
unsigned int max_cycle=1;

double G_drag_force_total;
double G_drag_force_pressure;
double G_drag_froce_viscous;

//Rheological model parameters
//Please Declare them in main function
double k_d;
double k_a;
double eta_str;
double eta_0;

namespace MyStokes
{
    using namespace dealii;
    
    template <int dim>
    class StokesProblem
    {
    public:
        StokesProblem (const unsigned int degree);
        void run ();
    
    private:
        
        void setup_dofs ();
        void assemble_system ();
        void solve_flow ();
        void solve_transport ();
        
        // assemble structure "lambda" transport equation
        void assemble_transport_system ();
        
        // "void assemble_transport_system ()" execute following subfunctions
        typedef MeshWorker::DoFInfo<dim> DoFInfo;
        typedef MeshWorker::IntegrationInfo<dim> CellInfo;
        static void integrate_cell_term (DoFInfo& dinfo, CellInfo& info);
        static void integrate_boundary_term (DoFInfo& dinfo, CellInfo& info);
        static void integrate_face_term (DoFInfo& dinfo1, DoFInfo& dinfo2,
                                         CellInfo& info1, CellInfo& info2);
        
        //Numerical solution of 'lambda' may overshoot [0.0, 1.0].
        //This should be regularized.
        void regularize_lambda ();
        
        void post_processing ();
        void refine_mesh (const unsigned int refinement_cycle);
        void compute_drag ();
        void output_results (const unsigned int refinement_cycle);
        
        const unsigned int   degree;
        Vector<double> cellwise_shear_rate;
        
        Triangulation<dim>   triangulation;
        FESystem<dim>        fe;
        MappingQ<dim>        mapping ;
        DoFHandler<dim>      dof_handler;
        ConstraintMatrix     constraints;
        BlockSparsityPattern      sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> solution;
        BlockVector<double> system_rhs;
        BlockVector<double> previous_solution;
        Vector<double> cellwise_errors;
   
        std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
    };
    
    //Class constructor
    template <int dim>
    StokesProblem<dim>::StokesProblem (const unsigned int degree)
    :
        degree (degree),
        triangulation (Triangulation<dim>::allow_anisotropic_smoothing),
        fe (FE_Q<dim>(degree+1), dim,
        FE_Q<dim>(degree), 1,
        FE_DGQ<dim>(degree), 1),
        mapping (degree),
        dof_handler (triangulation)
    {}
    
    
    template <int dim>
    void StokesProblem<dim>::setup_dofs ()
    {
        std::cout << "   -set_up_dof- allocating memory; start" <<std::endl;
    
        A_preconditioner.reset ();
        system_matrix.clear ();
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        
        std::vector<unsigned int> block_component (dim+2,0);
        block_component[dim] = 1;
        block_component[dim+1] = 2;
        
        DoFRenumbering::component_wise (dof_handler, block_component);
        std::vector<types::global_dof_index> dofs_per_block (3);
        DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1], n_s=dofs_per_block[2];
        
        std::cout << "   -n_u: " << n_u << "   -n_p: " << n_p <<"   -n_s: " << n_s << std::endl;
        
        {
            constraints.clear ();
            FEValuesExtractors::Vector velocities(0);
            FEValuesExtractors::Scalar radialvel(0);
            DoFTools::make_hanging_node_constraints (dof_handler,constraints);
            // Boundary ID 2 and 2 are symmetric-axis
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      2,
                                                      ZeroFunction<dim>(dim+2),
                                                      constraints,
                                                      fe.component_mask(radialvel));
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      4,
                                                      ZeroFunction<dim>(dim+2),
                                                      constraints,
                                                      fe.component_mask(radialvel));
            // Boundary ID 3 and 5 are the side-wall
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      3,
                                                      ConstantUz<dim>(),
                                                      constraints,
                                                      fe.component_mask(velocities));
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      5,
                                                      ConstantUz<dim>(),
                                                      constraints,
                                                      fe.component_mask(velocities));
            // Boundary ID 1 is the sphere
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      1,
                                                      ZeroFunction<dim>(dim+2),
                                                      constraints,
                                                      fe.component_mask(velocities));
        }
        
        constraints.close ();
        
        {
            BlockDynamicSparsityPattern dsp (3,3);
            
            dsp.block(0,0).reinit (n_u, n_u);
            dsp.block(1,0).reinit (n_p, n_u);
            dsp.block(2,0).reinit (n_s, n_u);
            dsp.block(0,1).reinit (n_u, n_p);
            dsp.block(1,1).reinit (n_p, n_p);
            dsp.block(2,1).reinit (n_s, n_p);
            dsp.block(0,2).reinit (n_u, n_s);
            dsp.block(1,2).reinit (n_p, n_s);
            dsp.block(2,2).reinit (n_s, n_s);
            
            dsp.collect_sizes();
            
            DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, true);
            DoFTools::make_flux_sparsity_pattern (dof_handler, dsp); // is this work well!?!?
            sparsity_pattern.copy_from (dsp);
            
        }
        
        system_matrix.reinit (sparsity_pattern);
        
        solution.reinit (3);
        solution.block(0).reinit (n_u);
        solution.block(1).reinit (n_p);
        solution.block(2).reinit (n_s);
        solution.collect_sizes ();
        
        {
            previous_solution.reinit (3);
            previous_solution.block(0).reinit (n_u);
            previous_solution.block(1).reinit (n_p);
            previous_solution.block(2).reinit (n_s);
            previous_solution.collect_sizes ();
        }
    
  
        system_rhs.reinit (3);
        system_rhs.block(0).reinit (n_u);
        system_rhs.block(1).reinit (n_p);
        system_rhs.block(2).reinit (n_s);
        system_rhs.collect_sizes ();
        
        std::cout << "   -set_up_dof" <<std::endl;
       
        std::cout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl;
        
    }
    
    
    template <int dim>
    void StokesProblem<dim>::assemble_system ()
    {
        std::cout << "   -begin assemble system" <<std::endl;
        
        system_matrix=0;
        system_rhs=0;
        
        QGauss<dim>   quadrature_formula(2*degree+1);
        
        FEValues<dim> fe_values (mapping, fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        const RightHandSide<dim>          right_hand_side;
        std::vector<Vector<double> >      rhs_values (n_q_points,
                                                      Vector<double>(dim+2));
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar radialvel (0);
        const FEValuesExtractors::Scalar pressure (dim);
        const FEValuesExtractors::Scalar structure (dim+1);
        
        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
        std::vector<double>                  div_phi_u   (dofs_per_cell);
        std::vector<double>                  phi_p       (dofs_per_cell);
        std::vector<double>                  phi_ur      (dofs_per_cell);
        
        std::vector<double>                  local_previous_solution_structure (n_q_points);
        
        double                               r_point;
        double                               lambda;
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            local_matrix = 0;
            local_rhs = 0;
            
            right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
            
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                r_point = fe_values.quadrature_point (q)[0]; // radial location
                
                //Extract Lambda from previous solution
                fe_values[structure].get_function_values (previous_solution, local_previous_solution_structure);
                double structure = local_previous_solution_structure[q];
                
                //Calculate local viscosity using structure solution
                double viscosity = viscosity_Moore (structure, eta_0, eta_str);
                
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                    div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                    phi_p[k]         = fe_values[pressure].value (k, q);
                    phi_ur[k]        = fe_values[radialvel].value (k, q);
                }
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j=0; j<=i; ++j)
                    {
                        local_matrix(i,j) += (2 * r_point * viscosity * (symgrad_phi_u[i] * symgrad_phi_u[j])
                                              + 2 * viscosity * phi_ur[i] * phi_ur[j] / r_point
                                              - (r_point * div_phi_u[i] + phi_ur[i]) * phi_p[j]
                                              - phi_p[i] * (r_point * div_phi_u[j] + phi_ur[j])
                                              + r_point * phi_p[i] * phi_p[j]
                                              )
                        * fe_values.JxW(q);
                        
                    }
        
                    const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                    local_rhs(i) += fe_values.shape_value(i,q) *
                    rhs_values[q](component_i) * r_point *
                    fe_values.JxW(q);
                }
            }
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i,j) = local_matrix(j,i);
            
            cell->get_dof_indices (local_dof_indices);
            
            constraints.distribute_local_to_global (local_matrix, local_rhs,
                                                    local_dof_indices,
                                                    system_matrix, system_rhs);
        }
        
        std::map<types::global_dof_index,double> boundary_values;
        
        A_preconditioner
        = std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type>(new typename InnerPreconditioner<dim>::type());
        A_preconditioner->initialize (system_matrix.block(0,0),
                                      typename InnerPreconditioner<dim>::type::AdditionalData());
    }
    
    template <int dim>
    void StokesProblem<dim>::assemble_transport_system ()
    {
        std::cout << "   -assemble transport equation" << std::endl;
        std::cout << "   -Setting up mesh worker ...\n";
    
        //deal.ii Mesh worker object is used
        MeshWorker::IntegrationInfoBox<dim> info_box;
        const unsigned int n_gauss_points = 2* dof_handler.get_fe().degree+1;
        info_box.initialize_gauss_quadrature(n_gauss_points,
                                             n_gauss_points,
                                             n_gauss_points);
        info_box.initialize_update_flags();
        UpdateFlags update_flags = update_quadrature_points |
                                   update_values|
                                   update_JxW_values|
                                   update_gradients;
        
        AnyData solution_data;
        solution_data.add<BlockVector<double>*>(&solution, "solution_numeric");
        info_box.cell_selector.add("solution_numeric", true, true, false );
        info_box.boundary_selector.add("solution_numeric", true, false, false );
        info_box.face_selector.add("solution_numeric", true, false, false );
        info_box.add_update_flags(update_flags, true, true, true, true);
    
        info_box.initialize (fe,mapping,solution_data,solution);
        
        MeshWorker::DoFInfo<dim> dof_info(dof_handler);
        MeshWorker::Assembler::SystemSimple<BlockSparseMatrix<double>, BlockVector<double>> assembler;
        assembler.initialize(system_matrix,system_rhs);
        
        MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim>>
        
        (dof_handler.begin_active(), dof_handler.end(),
         dof_info, info_box,
         &StokesProblem<dim>::integrate_cell_term,
         &StokesProblem<dim>::integrate_boundary_term,
         &StokesProblem<dim>::integrate_face_term,
         assembler);
        
    }
    
    template <int dim>
    void StokesProblem<dim>::integrate_cell_term (DoFInfo &dinfo,
                                                     CellInfo &info)
    {
        const FEValuesBase<dim> &fe_v = info.fe_values();
        const std::vector<double> &JxW = fe_v.get_JxW_values ();
        const std::vector<Point<2>> &quad_point = fe_v.get_quadrature_points ();
        const FEValuesExtractors::Scalar structure (dim+1);
        
        //extract previous solution
        //std::vector<Vector<double> > solution_values(fe_v.n_quadrature_points, Vector<double>(dim+2));
        const std::vector<double> &sol_ur = info.values[0][0]; //info.values[quadrature][component]
        const std::vector<double> &sol_uz = info.values[0][1]; //info.values[quadrature][component]
        // (dur/dx , dur/dy)
        const std::vector<Tensor<1,dim>> &local_grad_ur = info.gradients[0][0]; // info.values[quadrature][component]
        // (duz/dx , duz/dy)
        const std::vector<Tensor<1,dim>> &local_grad_uz = info.gradients[0][1]; // info.values[quadrature][component]
    
                //std::cout << "n_quadrature_points : " << fe_v.n_quadrature_points << std::endl;
        // Extract each component of gradient
        std::vector<SymmetricTensor<2,dim> > local_symgrad_u (1); //local_symgrad_u (how many?)
        // Usage local_symgrad_u[point][row][colum]
        
        double shear_rate =0;
        double r_point;
        double ur ;
        double dur_dr; double dur_dz; double duz_dr; double duz_dz;
     
        //This looks more complicate that it might seem necessary
        //This is because it can handle much more complex structures
        //than implemented here.
        FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
        Vector<double> &local_vector = dinfo.vector(0).block(0); 
    
        for (unsigned int q=0; q<fe_v.n_quadrature_points; ++q)
        {
            //extract values you need.
            Point<dim> advec;
            advec[0]=sol_ur[q];
            advec[1]=sol_uz[q];
            
            r_point = quad_point[q][0];
            ur = sol_ur[q];
            
            //caution......quiet dangerous to make mistake
            dur_dr = local_grad_ur[q][0]; dur_dz = local_grad_ur[q][1];
            duz_dr = local_grad_uz[q][0]; duz_dz = local_grad_uz[q][1];
            
            //make symmetric gradient
            local_symgrad_u[0][0][0]= dur_dr;
            local_symgrad_u[0][0][1]= 0.5*(dur_dz+duz_dr);
            local_symgrad_u[0][1][0]= 0.5*(dur_dz+duz_dr);
            local_symgrad_u[0][1][1]= duz_dz;
            
            shear_rate = get_shear_rate(local_symgrad_u[0],ur,r_point);
            // ur is correct (checked sum...) // r_point is also correct //shear_rate value is correct, checked.
            
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            {   const Tensor<1,dim> grad_phi_i_s = fe_v[structure].gradient (i, q);
                const double phi_i_s = fe_v[structure].value(i,q);
                
                 for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                 {
                    const double phi_j_s = fe_v[structure].value(j,q);
                    local_matrix(i,j) += ( (-1)*advec *phi_j_s* grad_phi_i_s
                                         + (k_a+ shear_rate * k_d) * phi_j_s * phi_i_s) *
                                            r_point *
                                            JxW[q]; // - (lambda, u \cdot grad w)
                 }
                    local_vector(i) += k_a*
                                       phi_i_s  *
                                       r_point *
                                       JxW[q];
            }
        } //close quadrature_cycle
    }
    
    template <int dim>
    void StokesProblem<dim>::integrate_boundary_term (DoFInfo &dinfo,
                                                      CellInfo &info)
    {
        const FEValuesExtractors::Scalar structure (dim+1);
        const FEValuesBase<dim> &fe_v = info.fe_values();
        
        FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
        Vector<double> &local_vector = dinfo.vector(0).block(0);
        const std::vector<double> &JxW = fe_v.get_JxW_values ();
        
        const std::vector<Point<2>> &quad_point = fe_v.get_quadrature_points ();
        
        //extract previous solution
        //std::vector<Vector<double> > solution_values(fe_v.n_quadrature_points, Vector<double>(dim+2));
        const std::vector<double> &sol_ur = info.values[0][0]; //info.values[quadrature][component]
        const std::vector<double> &sol_uz = info.values[0][1]; //info.values[quadrature][component]
        
        const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();
        
        static StructureBoundaryValues<dim> boundary_function;
        std::vector<double> boundary_values(fe_v.n_quadrature_points);
        boundary_function.value_list (fe_v.get_quadrature_points(), boundary_values);
        
        double r_point;
        
        for (unsigned int q=0; q<fe_v.n_quadrature_points; ++q)
         {
             // x_i
             r_point = quad_point[q][0];
             Point<dim> advec;
             advec[0]=sol_ur[q]; advec[1]=sol_uz[q];
             
             double u_norm = std::sqrt(advec[0] * advec[0] + advec[1] * advec[1]);
             const double advec_n=advec * normals[q]; //
    
             if ( advec_n <0 && quad_point[q][0] !=1)
                    {
                    for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                            local_vector(i) -= advec_n *
                            boundary_values[q] *
                            fe_v.shape_value(i,q) *
                            r_point *
                            JxW[q];
                    }
                else
                    {
                    for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                        for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                            local_matrix(i,j) += advec_n *
                            fe_v[structure].value(i,q) *
                            fe_v[structure].value(j,q) *
                            r_point *
                            JxW[q];
                    }
         }
        
    }
    
    
    template <int dim>
    void StokesProblem<dim>::integrate_face_term (DoFInfo &dinfo1,
                                                     DoFInfo &dinfo2,
                                                     CellInfo &info1,
                                                     CellInfo &info2)
    {
        const FEValuesExtractors::Scalar structure (dim+1);
        const FEValuesBase<dim> &fe_v = info1.fe_values();
        const FEValuesBase<dim> &fe_v_neighbor = info2.fe_values();
        FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0,false).matrix;
        FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0,true).matrix;
        FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0,true).matrix;
        FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0,false).matrix;
        const std::vector<double> &JxW = fe_v.get_JxW_values ();
        const std::vector<Point<2>> &quad_point = fe_v.get_quadrature_points ();
        const std::vector<double> &sol_ur = info1.values[0][0]; //info.values[quadrature][component]
        const std::vector<double> &sol_uz = info1.values[0][1]; //info.values[quadrature][component]
        
        const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();
        
        double r_point;
        
        for (unsigned int q=0; q<fe_v.n_quadrature_points; ++q)
        {
            // x_i
            r_point = quad_point[q][0];
            
            //v -Solutions
            Point<dim> advec;
            advec[0]=sol_ur[q];
            advec[1]=sol_uz[q];
            
            const double advec_n=advec * normals[q]; //
            
            if (advec_n>0)
            {
                for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                    for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                        u1_v1_matrix(i,j) += advec_n *
                        fe_v[structure].value(j,q) *
                        fe_v[structure].value(i,q) *
                        r_point *
                        JxW[q];
                
                for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
                    for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                        u1_v2_matrix(k,j) -= advec_n *
                        fe_v[structure].value(j,q) *
                        fe_v_neighbor[structure].value(k,q) *
                        r_point *
                        JxW[q];
            }
            
            else
            {
                for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                    for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                        u2_v1_matrix(i,l) += advec_n *
                        fe_v_neighbor[structure].value(l,q) *
                        fe_v[structure].value(i,q) *
                        r_point *
                        JxW[q];
                
                for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
                    for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                        u2_v2_matrix(k,l) -= advec_n *
                        fe_v_neighbor[structure].value(l,q) *
                        fe_v_neighbor[structure].value(k,q) *
                        r_point *
                        JxW[q];
            }
        }
        
    }

    
    template <int dim>
    void StokesProblem<dim>::solve_flow ()
    {
        const InverseMatrix<SparseMatrix<double>,
        typename InnerPreconditioner<dim>::type>
        A_inverse (system_matrix.block(0,0), *A_preconditioner);
        
        Vector<double> tmp (solution.block(0).size());
        
        {
            Vector<double> schur_rhs (solution.block(1).size());
            A_inverse.vmult (tmp, system_rhs.block(0));
            system_matrix.block(1,0).vmult (schur_rhs, tmp);
            schur_rhs -= system_rhs.block(1);
            
            SchurComplement<typename InnerPreconditioner<dim>::type>
            schur_complement (system_matrix, A_inverse);
            
            SolverControl solver_control ( 10 * solution.block(1).size(),
                                          1e-10*schur_rhs.l2_norm());
            SolverCG<>    cg (solver_control);
            
            SparseILU<double> preconditioner;
            preconditioner.initialize (system_matrix.block(1,1),
                                       SparseILU<double>::AdditionalData());
            
            InverseMatrix<SparseMatrix<double>,SparseILU<double> >
            m_inverse (system_matrix.block(1,1), preconditioner);
            
            cg.solve (schur_complement, solution.block(1), schur_rhs,
                      m_inverse);
            
            constraints.distribute (solution);
            
            std::cout << "   -"
            << solver_control.last_step()
            << " outer CG Schur complement iterations for pressure"
            << std::endl;
        }
        
        
        {
            system_matrix.block(0,1).vmult (tmp, solution.block(1));
            tmp *= -1;
            tmp += system_rhs.block(0);
            
            A_inverse.vmult (solution.block(0), tmp);
            
            constraints.distribute (solution);
        }
    }
    
    
    template <int dim>
    void StokesProblem<dim>::solve_transport ()
    {
        std::cout << "   -solve transport begins"<< std::endl;
        SolverControl           solver_control (std::pow(10,6), system_rhs.block(2).l2_norm() * pow(10,-4));
        
        unsigned int restart = 500;
        SolverGMRES< Vector<double> >::AdditionalData gmres_additional_data(restart+2);
        SolverGMRES< Vector<double> > solver(solver_control, gmres_additional_data);
      
        //make preconditioner
        SparseILU<double>::AdditionalData additional_data(0,500); // (0 , additional diagonal terms)
        SparseILU<double> preconditioner;
        preconditioner.initialize (system_matrix.block(2,2), additional_data);
        solver.solve (system_matrix.block(2,2), solution.block(2), system_rhs.block(2), preconditioner);
      
    }
    
 
    template <int dim>
    void
    StokesProblem<dim>::output_results (const unsigned int refinement_cycle)
    {
        std::vector<std::string> solution_names (dim, "Velocity");
        solution_names.push_back ("Pressure");
        solution_names.push_back ("Structure");
            
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, solution_names,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);
            
        data_out.build_patches ();
            
        std::ostringstream filenameeps;
        filenameeps << "a0_25_U"<< U_inflow<<"_kd"<<k_d<<".plt";
            
        std::ofstream output (filenameeps.str().c_str());
        data_out.write_tecplot(output);
    }
    
 
    template <int dim>
    void
    StokesProblem<dim>::refine_mesh (const unsigned int refinement_cycle)
    {
        
        if(refinement_cycle <= 6)
        {
            GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                             cellwise_shear_rate,
                                                             0.3, 0.0);
            // 30% Cells will be refined and 0% Cell will be coarsened
        }
        
        else if(refinement_cycle>6 && refinement_cycle<9)
        {
            GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                             cellwise_shear_rate,
                                                             0.3, 0.0);
        }
        
        else
        {
            GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                             cellwise_shear_rate,
                                                             0.15, 0.0);
        }
        
        //Solution from coarsed mesh will be transferred to refined mesh
        //to guarantee faster iteration with the new mesh
        SolutionTransfer<dim,BlockVector<double>> solution_transfer(dof_handler);
        triangulation.prepare_coarsening_and_refinement ();
        solution_transfer.prepare_for_coarsening_and_refinement(solution);
        triangulation.execute_coarsening_and_refinement ();
        
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        std::vector<unsigned int> block_component (dim+2,0);
        block_component[dim] = 1;
        block_component[dim+1] = 2;
        DoFRenumbering::component_wise (dof_handler, block_component);
        
        triangulation.execute_coarsening_and_refinement ();
        std::vector<types::global_dof_index> dofs_per_block (3);
        DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1], n_s=dofs_per_block[2];
        
        // save new space for previous solution,
        // you need larger memory than previous mesh
        previous_solution.reinit (3);
        previous_solution.block(0).reinit (n_u);
        previous_solution.block(1).reinit (n_p);
        previous_solution.block(2).reinit (n_s);
        previous_solution.collect_sizes ();
   
        solution_transfer.interpolate(solution, previous_solution);
        
    }
    
    
    template <int dim>
    void
    StokesProblem<dim>::compute_drag ()
    {
        std::cout << "  Compute drag..." << std::endl;
        
        const long double pi = 3.141592653589793238462643;
        const MappingQ<dim> mapping (degree);
        double r_point;
        double viscous_drag=0;
        double pressure_drag=0;
        double total_drag =0;
        
        QGauss<dim-1>   quadrature_formula_face(2*degree+1); //check weather this is enough
        
        FEFaceValues<dim> fe_face_values (mapping, fe, quadrature_formula_face,
                                          update_JxW_values |
                                          update_quadrature_points |
                                          update_gradients |
                                          update_values |
                                          update_normal_vectors);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar radialvel (0);
        const FEValuesExtractors::Scalar pressure (dim);
        const FEValuesExtractors::Scalar structure (dim+1);
        
        const unsigned int   faces_per_cell  = GeometryInfo<dim>::faces_per_cell;
        const unsigned int   n_q_face_points = fe_face_values.n_quadrature_points;
        
        std::vector<double>                  local_pressure_values (n_q_face_points);
        std::vector<double>                  local_ur_values (n_q_face_points);
        std::vector<double>                  local_structure_values (n_q_face_points);
        // extract structure value to calculate local viscosity
        
        std::vector<SymmetricTensor<2,dim>>  local_sym_vel_gradient (n_q_face_points);
        Tensor<1,dim>                        normal;
        
        //values at each quadrature point
        double lambda_s;
        double viscosity;
        double shear_rate;
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            
            for (unsigned int face_no=0; face_no<faces_per_cell; ++face_no)
                if (cell->face(face_no)->boundary_id()==1)
                {
                    
                    fe_face_values.reinit (cell, face_no);
                    fe_face_values[pressure].get_function_values (solution,local_pressure_values);
                    fe_face_values[velocities].get_function_symmetric_gradients (solution,local_sym_vel_gradient);
                    fe_face_values[radialvel].get_function_values (solution, local_ur_values);
                    fe_face_values[structure].get_function_values (solution, local_structure_values);
                    
                    for (unsigned int q=0; q<n_q_face_points; ++q)
                    {
                        lambda_s = local_structure_values [q];
                        viscosity = viscosity_Moore (lambda_s, eta_0, eta_str);
                        r_point = fe_face_values.quadrature_point (q)[0];
                        normal = fe_face_values.normal_vector (q);
                        shear_rate = get_shear_rate(local_sym_vel_gradient[q], local_ur_values[q], r_point);
                        pressure_drag += 2.* pi * r_point  * (normal[1]*local_pressure_values[q]) * fe_face_values.JxW (q) ;
                        viscous_drag += 2.* pi * r_point * (-2.*viscosity*normal[0]*local_sym_vel_gradient[q][0][1] +
                                                            -2.*viscosity*normal[1]*local_sym_vel_gradient[q][1][1]
                                                            )*fe_face_values.JxW (q);
                        
                    }
                }
            
        }
        
        
        total_drag= pressure_drag + viscous_drag;
        
        std::cout << std::fixed << std::setprecision(10) <<
        "   DRAG = " << total_drag << "    " << "Pressure Drag = " << pressure_drag
        << "   " <<"Viscous Drag = " << viscous_drag <<"   " << std::endl ;
        
        G_drag_force_total=total_drag;
        G_drag_force_pressure=pressure_drag;
        G_drag_froce_viscous=viscous_drag;
        
    }
    
    template <int dim>
    void StokesProblem<dim>::regularize_lambda ()
    {
        std::vector<unsigned int> block_component (dim+2,0);
        block_component[dim] = 1;
        block_component[dim+1] = 2;
        
        std::vector<types::global_dof_index> dofs_per_block (3);
        
        DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const unsigned int n_s=dofs_per_block[2];
        for  (unsigned int i = 0; i<n_s; ++i)
        {
            if(solution.block(2)[i]<0)
                solution.block(2)[i]=0;
        }
        
        
    }
    
    //This is an auxilliary function. For analysis only,if one is interested.
    template <int dim>
    void StokesProblem<dim>::post_processing ()
    {
        const MappingQ<dim> mapping (degree);
        
        double r_point; // Raidal position. Plays important role in any integration
        cellwise_shear_rate.reinit (triangulation.n_active_cells());
        QGauss<dim>   quadrature_formula(degree+2);
        FEValues<dim> fe_values (mapping, fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        std::vector<SymmetricTensor<2,dim> > local_symgrad_phi_u (n_q_points);
        std::vector<double>                  local_phi_ur (n_q_points);
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar radialvel (0);
        const FEValuesExtractors::Scalar pressure (dim);
        //Part 1 : Calculate Stress Field on each cell
        unsigned int k=0;
    
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            fe_values[velocities].get_function_symmetric_gradients (solution, local_symgrad_phi_u);
            fe_values[radialvel].get_function_values (solution, local_phi_ur);
            cellwise_shear_rate(k)=0;
            
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                r_point = fe_values.quadrature_point (q)[0]; // radial location
                double shear_rate = std::sqrt( (local_symgrad_phi_u[q]* local_symgrad_phi_u[q]
                                                + pow((local_phi_ur[q]/r_point),2) ) *2 );
                cellwise_shear_rate(k)+=shear_rate;
                
            }
                cellwise_shear_rate(k)=cellwise_shear_rate(k)/n_q_points;
                k+=1;
        }
        
    }
    
    template <int dim>
    void StokesProblem<dim>::run ()
    {
        {   //Read Mesh
            std::vector<unsigned int> subdivisions (dim, 1);
            subdivisions[0] = 4;
            GridIn<dim> grid_in;
            grid_in.attach_triangulation (triangulation);
            std::ifstream input_file("MESH_mean.inp");
            Assert (dim==2, ExcInternalError());
            grid_in.read_ucd (input_file);
            const Point<2> center (0,0);
            static const SphericalManifold<dim> manifold_description;
            triangulation.set_manifold (1, manifold_description);
            
        }
        
        for (unsigned int refinement_cycle = 0; refinement_cycle<3; ++refinement_cycle)
        {
            std::cout << "Refinement cycle " << refinement_cycle << std::endl;
            
            if (refinement_cycle==0)
                triangulation.refine_global (2);
            else
                refine_mesh(refinement_cycle);
            
            setup_dofs ();
            assemble_system ();
            solve_flow ();
            assemble_transport_system ();
            solve_transport ();
            regularize_lambda ();
            
            BlockVector<double> difference;
            int iteration_number=0 ;
            
            previous_solution =solution ;
            
            do{
                iteration_number +=1;
                
                assemble_system ();
                solve_flow ();
                assemble_transport_system ();
                solve_transport ();
                
                regularize_lambda ();
                
                difference = solution;
                difference -= previous_solution;
                previous_solution=solution;
                
                std::cout << "   Iteration Number showing : " << iteration_number << "     Difference Norm : " << difference.l2_norm() << std::endl << std::flush;
                
            }while (difference.l2_norm()>pow(10,-6)* dof_handler.n_dofs());
            
            post_processing ();
            
            compute_drag ();
            output_results (refinement_cycle);
            
            
        }
            output_results (1);
    }
    
   
}

int main ()
{
    try
    {
        using namespace dealii;
        using namespace MyStokes;
    
        U_inflow=0.1;
        k_d=0.5;
        k_a=1.0;
        eta_str=2;
        eta_0=1.0;
        
        StokesProblem<2> flow_problem1(1);
        flow_problem1.run ();
        
        
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    
    return 0;
}
