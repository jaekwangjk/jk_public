// Generalized Newtonian Stokes Flow Solver
// based on Finite Element Method
// Written by Jaekwang Kim
// This code has been used for research UQ Analysis of GNF flow simluation
// and publication related to this is JNNFM(2019) Vol 271, 104138
// "Uncertainty propagation in simulation predictions
// of generalized Newtonian fluid flows"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
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
//for multi-threading
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
//include header files
#include "headers/bc_and_rhs.h"
#include "headers/precondition.h"
#include "headers/post_processor.h"

#define Pi 3.141592653589793238462643;


double max_cycle=7;

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
    
        //Basic Structure
        void setup_dofs (const unsigned int refinement_cycle);
        void solve (); // Matrix Solver based on CG algorithm
        void output_results (const unsigned int refinement_cycle) const;
        void refine_mesh (const unsigned int refinement_cycle, double max_shear);
        void set_anisotropic_flags ();
        void post_processing (); //Calculate shear-rate field and stress-field
        void compute_drag ();
    
        double get_shear_rate (const SymmetricTensor<2,dim> symgrad_u, const double ur, const double r_point);
        
        //For multi threading
        struct AssemblyScratchData
        {
            AssemblyScratchData (const FESystem<dim> &fe);
            AssemblyScratchData (const AssemblyScratchData &scratch_data);
            FEValues<dim>     fe_values;
        };
        struct AssemblyCopyData
        {
            FullMatrix<double>                   local_matrix;
            Vector<double>                       local_rhs;
            std::vector<types::global_dof_index> local_dof_indices;
        };
        
        void local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    AssemblyScratchData                                  &scratch,
                                    AssemblyCopyData                                     &copy_data);
        void copy_local_to_global (const AssemblyCopyData &copy_data);
        void assemble_system();
        const unsigned int   degree;
        
        // Varibles for iterative scheme /
        BlockVector<double> previous_solution;
        // Variables to save the result from 'post_processing' function
        Vector<double> cellwise_shear_rate;
        Vector<double> cellwise_stress_field;
        Vector<double> cellwise_viscosity;
        
        // GLobal Variables
        Triangulation<dim>   triangulation;
        FESystem<dim>        fe;
        DoFHandler<dim>      dof_handler;
        ConstraintMatrix     constraints;
        BlockSparsityPattern      sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        BlockVector<double> solution;
        BlockVector<double> system_rhs;
        
        //Herschel Berkeley Model parameter - For Constitutitive Equation
        
        double K=2.89;
        double tau_y= 5.99;
        double power_n=0.4057;
        double m =pow(10,6); //Papanastasiou regularization
        
        double max_shear_rate=1; //init. arbitray constant
        double max_area;
        long double min_area;
        double area_ratio=1;
        std_cxx11::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
        
    };
    
    template <int dim>
    StokesProblem<dim>::StokesProblem (const unsigned int degree)
    :
    degree (degree),
    triangulation (Triangulation<dim>::allow_anisotropic_smoothing),
    fe (FE_Q<dim>(degree+1), dim,
        FE_Q<dim>(degree), 1),
    dof_handler (triangulation){}
    
    template<int dim>
    double StokesProblem<dim>::get_shear_rate(const SymmetricTensor<2,dim> symgrad_u,
                                              const double ur, const double r_point)
    {
        return std::sqrt( 2.*(symgrad_u*symgrad_u + ur*ur/(r_point*r_point) ) );
    }
    
    template <int dim>
    void StokesProblem<dim>::setup_dofs (const unsigned int refinement_cycle)
    {
        
        A_preconditioner.reset ();
        system_matrix.clear ();
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler, block_component);
        
        {
            constraints.clear ();
            
            FEValuesExtractors::Vector velocities(0);
            FEValuesExtractors::Scalar radialvel(0);
            
            DoFTools::make_hanging_node_constraints (dof_handler,
                                                     constraints);
            
            // AXIS - NO PEN boundary_id=2 and 4
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      2,
                                                      ZeroFunction<dim>(dim+1),
                                                      constraints,
                                                      fe.component_mask(radialvel));
            
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      4,
                                                      ZeroFunction<dim>(dim+1),
                                                      constraints,
                                                      fe.component_mask(radialvel));
        
            // TOP & Bottom Boundary , boundary_id =3
            // Outer Wall, boundary_id=5
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
            
            // Sphere Boundary boundary id=1
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      1,
                                                      ZeroFunction<dim>(dim+1),
                                                      constraints,
                                                      fe.component_mask(velocities));
        }
        
        constraints.close ();
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];
        
        std::cout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << " (" << n_u << '+' << n_p << ')'
        << std::endl;
        
        {
            BlockDynamicSparsityPattern dsp (2,2);
            
            dsp.block(0,0).reinit (n_u, n_u);
            dsp.block(1,0).reinit (n_p, n_u);
            dsp.block(0,1).reinit (n_u, n_p);
            dsp.block(1,1).reinit (n_p, n_p);
            
            dsp.collect_sizes();
            
            DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
            sparsity_pattern.copy_from (dsp);
        }
        
        system_matrix.reinit (sparsity_pattern);
        
        solution.reinit (2);
        solution.block(0).reinit (n_u);
        solution.block(1).reinit (n_p);
        solution.collect_sizes ();
        
        if(refinement_cycle==0)
        {
            previous_solution.reinit (2);
            previous_solution.block(0).reinit (n_u);
            previous_solution.block(1).reinit (n_p);
            previous_solution.collect_sizes ();
        }
        
        
        system_rhs.reinit (2);
        system_rhs.block(0).reinit (n_u);
        system_rhs.block(1).reinit (n_p);
        system_rhs.collect_sizes ();
    }
    
  
    template <int dim>
    void StokesProblem<dim>::assemble_system ()
    {
        system_matrix=0;
        system_rhs=0;
        
        WorkStream::run(dof_handler.begin_active(),
                        dof_handler.end(),
                        *this,
                        &StokesProblem::local_assemble_system,
                        &StokesProblem::copy_local_to_global,
                        AssemblyScratchData(fe),
                        AssemblyCopyData());
        
        A_preconditioner
        = std::shared_ptr<typename InnerPreconditioner<dim>::type>(new typename InnerPreconditioner<dim>::type());
        A_preconditioner->initialize (system_matrix.block(0,0),
                                      typename InnerPreconditioner<dim>::type::AdditionalData());
    }
    
    template <int dim>
    StokesProblem<dim>::AssemblyScratchData::
    AssemblyScratchData (const FESystem<dim> &fe)
    :
    fe_values (fe,
               QGauss<dim>(3),  // Hard coded.
               update_values   | update_gradients |
               update_quadrature_points | update_JxW_values)
    {}
    
    template <int dim>
    StokesProblem<dim>::AssemblyScratchData::
    AssemblyScratchData (const AssemblyScratchData &scratch_data)
    :
    fe_values (scratch_data.fe_values.get_fe(),
               scratch_data.fe_values.get_quadrature(),
               update_values   | update_gradients |
               update_quadrature_points | update_JxW_values)
    {}
    
    
    template <int dim>
    void
    StokesProblem<dim>::
    local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                           AssemblyScratchData                                  &scratch_data,
                           AssemblyCopyData                                     &copy_data)
    {
        
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
        
        copy_data.local_matrix.reinit (dofs_per_cell, dofs_per_cell);
        copy_data.local_rhs.reinit (dofs_per_cell);
        copy_data.local_dof_indices.resize(dofs_per_cell);
        
        const RightHandSide<dim>  right_hand_side;
        std::vector<Vector<double> >      rhs_values (n_q_points,Vector<double>(dim+1));
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar radialvel (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
        std::vector<double>                  div_phi_u   (dofs_per_cell);
        std::vector<double>                  phi_p       (dofs_per_cell);
        std::vector<double>                  phi_ur      (dofs_per_cell);
        
        //Variables from previous solution
        std::vector<SymmetricTensor<2,dim> > local_previous_symgrad_phi_u (n_q_points);
        std::vector<double>                  local_previous_solution (n_q_points);
        std::vector<double>                  local_phi_ur_previous (n_q_points);
        
        double r_point;
        double viscosity;
        double shear_rate;
        
        scratch_data.fe_values.reinit (cell);
        right_hand_side.vector_value_list(scratch_data.fe_values.get_quadrature_points(), rhs_values);
        
        
        scratch_data.fe_values[velocities].get_function_symmetric_gradients
        (previous_solution, local_previous_symgrad_phi_u);
        scratch_data.fe_values[radialvel].get_function_values
        (previous_solution, local_phi_ur_previous);
        
        for (unsigned int q=0; q<n_q_points; ++q)
            
        {
            r_point = scratch_data.fe_values.quadrature_point (q)[0];
            shear_rate = get_shear_rate(local_previous_symgrad_phi_u[q],
                                        local_phi_ur_previous[q], r_point);
            
            if(shear_rate ==0 )
            {
                viscosity = 1;
            }else{
                viscosity = K * pow(shear_rate,power_n-1) +
                tau_y*(1-exp(0-m*shear_rate))/(shear_rate);
            }
            
    
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
                symgrad_phi_u[k] = scratch_data.fe_values[velocities].symmetric_gradient (k, q);
                div_phi_u[k]     = scratch_data.fe_values[velocities].divergence (k, q);
                phi_p[k]         = scratch_data.fe_values[pressure].value (k, q);
                phi_ur[k]        = scratch_data.fe_values[radialvel].value (k, q);
            }
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    copy_data.local_matrix(i,j) += (2 * r_point * viscosity * (symgrad_phi_u[i]
                                                                               * symgrad_phi_u[j])
                                                    + 2 * viscosity * phi_ur[i] * phi_ur[j] / r_point
                                                    - (r_point * div_phi_u[i] + phi_ur[i]) * phi_p[j]
                                                    - phi_p[i] * (r_point * div_phi_u[j] + phi_ur[j])
                                                    + r_point * phi_p[i] * phi_p[j]
                                                    )
                    * scratch_data.fe_values.JxW(q);
                }
                
                const unsigned int component_i =fe.system_to_component_index(i).first;
        
                copy_data.local_rhs(i) += scratch_data.fe_values.shape_value(i,q) *
                rhs_values[q](component_i) * r_point *
                scratch_data.fe_values.JxW(q);
            }
        }
        cell->get_dof_indices (copy_data.local_dof_indices);
    }
    
    
    template <int dim>
    void
    StokesProblem<dim>::copy_local_to_global (const AssemblyCopyData &copy_data)
    {
        constraints.distribute_local_to_global (copy_data.local_matrix, copy_data.local_rhs,
                                                copy_data.local_dof_indices,
                                                system_matrix, system_rhs);
    }
    
    
    template <int dim>
    void StokesProblem<dim>::solve ()
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
            
            SolverControl solver_control ( 1000 * solution.block(1).size(),
                                          1e-5*schur_rhs.l2_norm());
            SolverCG<>    cg (solver_control);
            
            SparseILU<double> preconditioner;
            preconditioner.initialize (system_matrix.block(1,1),
                                       SparseILU<double>::AdditionalData());
            
            InverseMatrix<SparseMatrix<double>,SparseILU<double> >
            m_inverse (system_matrix.block(1,1), preconditioner);
            
            cg.solve (schur_complement, solution.block(1), schur_rhs,
                      m_inverse);
            
            constraints.distribute (solution);
            
            std::cout << "  "
            << solver_control.last_step()
            << "  outer CG Schur complement iterations for pressure"
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
    void
    StokesProblem<dim>::output_results (const unsigned int refinement_cycle)  const
    {
        
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.push_back ("pressure");
        
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
    
        ShearRate<dim> shear_rate; // This should be declared before Dataout Class
        DataOut<dim> data_out;
        
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, solution_names,
                                  DataOut<dim>::type_dof_data,
                                  data_component_interpretation);
        
        //shear rate is calculated from post_processor
        data_out.add_data_vector (solution, shear_rate);
        data_out.add_data_vector (cellwise_shear_rate,"Shear_rate");
        data_out.build_patches ();
        
        
        std::ostringstream filenameeps;
        filenameeps << "solution-"<< Utilities::int_to_string (refinement_cycle, 2)<< ".vtk";
        
        std::ofstream output (filenameeps.str().c_str());
        data_out.write_vtk (output);
        
    }
    
    
    
    
    template <int dim>
    void
    StokesProblem<dim>::refine_mesh (const unsigned int refinement_cycle, double max_shear)
    //Adaptive Mesh Refinement - criteria is shear-rate
    //Also, I conduct anistoropy mesh-refinement, after 3th refinement
    {
        
        if(refinement_cycle <= 3)
        {
            GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                             cellwise_shear_rate,
                                                             0.5, 0.0);
        }
        
        else if(refinement_cycle>4 && refinement_cycle<9)
        {
            GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                             cellwise_shear_rate,
                                                             0.3, 0.0);
            set_anisotropic_flags();
        }
        
        else
        {
            GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                             cellwise_shear_rate,
                                                             0.15, 0.0);
            set_anisotropic_flags();
        }
        
        
        SolutionTransfer<dim,BlockVector<double>> solution_transfer(dof_handler);
        //Prepare Solution Transfer for Block vector solution
        //Solution Transfer from coarsed mesh to refined mesh accelrates computational speed remarkably
        
        triangulation.prepare_coarsening_and_refinement ();
        solution_transfer.prepare_for_coarsening_and_refinement(solution);
        triangulation.execute_coarsening_and_refinement ();
        
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::Cuthill_McKee (dof_handler);
        
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler, block_component);
        
        triangulation.execute_coarsening_and_refinement ();
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];
        
        // Save new memory space for previous solution, you need larger memory than previous mesh
        previous_solution.reinit (2);
        previous_solution.block(0).reinit (n_u);
        previous_solution.block(1).reinit (n_p);
        previous_solution.collect_sizes ();
        
        // Transfer solution
        solution_transfer.interpolate(solution, previous_solution);
        
    }

    // For anistopic refinment
    template <int dim>
    void
    StokesProblem<dim>::set_anisotropic_flags ()
    {
        typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active()
        ,endc=dof_handler.end();
        
        for (; cell!=endc; ++cell)
        {
            if (cell->refine_flag_set())
            {
                cell->set_refine_flag(RefinementCase<dim>::cut_y);
            }
        }
    }
    
    //Calculate stress field from solution (u,p) and viscosity and shear-rate
    template <int dim>
    void
    StokesProblem<dim>::post_processing ()
    {
        max_shear_rate = 0;
        double temp_shear =0 ;
        
        max_area=0;
        min_area=1000;
        
        const MappingQ<dim> mapping (degree);
        double r_point;
        
        cellwise_shear_rate.reinit (triangulation.n_active_cells());
        cellwise_stress_field.reinit (triangulation.n_active_cells());
        cellwise_viscosity.reinit (triangulation.n_active_cells());
        
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
            
            
            double area=0;
            
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                
                r_point = fe_values.quadrature_point (q)[0]; // radial location
                double shear_rate = std::sqrt( (local_symgrad_phi_u[q]* local_symgrad_phi_u[q]
                                                + pow((local_phi_ur[q]/r_point),2) ) *2 );
                
                cellwise_shear_rate(k)+=shear_rate;
                
                area+=fe_values.JxW(q);
                
            }
            
            cellwise_shear_rate(k)=cellwise_shear_rate(k)/n_q_points;
            
            temp_shear=cellwise_shear_rate(k);
            
            
            if(temp_shear>max_shear_rate)
            {
                max_shear_rate=temp_shear;
            }
            
            
            if(area>max_area)
            {
                max_area=area;
            }
            
            if(area<min_area)
            {
                min_area=area;
            }
            
            
            cellwise_viscosity(k)=K * pow(cellwise_shear_rate(k),power_n-1) +
            tau_y*(1-exp(0-m*cellwise_shear_rate(k)))/(cellwise_shear_rate(k));
            cellwise_stress_field(k)= cellwise_viscosity(k) * cellwise_shear_rate(k) ;
            
            
            
            k+=1;
            
            
            
        }
        
        area_ratio=max_area/min_area;
        
        
    }
    
    
    
    // Calculate Drag force on sphere
    template <int dim>
    void
    StokesProblem<dim>::compute_drag ()
    {
        const long double pi = 3.141592653589793238462643;
        
        const MappingQ<dim> mapping (degree);
        
        double r_point;
        
        double viscous_drag=0;
        double pressure_drag=0;
        double total_drag =0;
        
        QGauss<dim-1>   quadrature_formula_face(2*degree+1);
        
        FEFaceValues<dim> fe_face_values (mapping, fe, quadrature_formula_face,
                                          update_JxW_values |
                                          update_quadrature_points |
                                          update_gradients |
                                          update_values |
                                          update_normal_vectors);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar radialvel (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        //Drag will be calculated from faces of cell, not any values inside the cell
        const unsigned int   faces_per_cell  = GeometryInfo<dim>::faces_per_cell;
        const unsigned int   n_q_face_points = fe_face_values.n_quadrature_points;
        
        std::vector<double>                  local_pressure_values (n_q_face_points);
        std::vector<double>                  local_ur_values (n_q_face_points);
        std::vector<SymmetricTensor<2,dim>>  local_sym_vel_gradient (n_q_face_points);
        Tensor<1,dim>                        normal;
        
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
                    //You have to calculate visosity at that point
                    
                    fe_face_values.reinit (cell, face_no);
                    fe_face_values[pressure].get_function_values (solution,local_pressure_values);
                    fe_face_values[velocities].get_function_symmetric_gradients (solution,local_sym_vel_gradient);
                    fe_face_values[radialvel].get_function_values (solution, local_ur_values);
                    
                    for (unsigned int q=0; q<n_q_face_points; ++q)
                    {
                        
                    r_point = fe_face_values.quadrature_point (q)[0];
                    normal = fe_face_values.normal_vector (q);
                        
                    shear_rate = get_shear_rate(local_sym_vel_gradient[q], local_ur_values[q], r_point);
                        
                    viscosity = K * pow(shear_rate,power_n-1) + tau_y*(1-exp(0-m*shear_rate))/(shear_rate);
                        
                    pressure_drag += 2.* pi * r_point  * (normal[1]*local_pressure_values[q]) * fe_face_values.JxW (q) ;
                        
                    viscous_drag += 2.* pi * r_point * (-2.*viscosity*normal[0]*local_sym_vel_gradient[q][0][1] +
                                                            -2.*viscosity*normal[1]*local_sym_vel_gradient[q][1][1]
                                                            )*fe_face_values.JxW (q);
                        
                    }
                }
            
        }
        
        
        total_drag= pressure_drag + viscous_drag;
        
        std::cout << std::fixed << std::setprecision(10) <<
        "   DRAG = " << total_drag << "    " << "Pressure Drag = " << pressure_drag << "   " <<"Viscous Drag = " << viscous_drag <<"   " << std::endl ;
        
        G_drag_force_n=total_drag;
        
        std::ofstream output_mesh("mesh_drag.dat",std::ios::app);
        output_mesh << triangulation.n_active_cells() << " " << G_drag_force_n<< std::endl;
        
    }
    
    
    template <int dim>
    void StokesProblem<dim>::run ()
    {
        //Read Mesh
        {
            std::vector<unsigned int> subdivisions (dim, 1);
            subdivisions[0] = 4;
            
            GridIn<dim> grid_in;
            grid_in.attach_triangulation (triangulation);
            
            //real mesh
            std::ifstream input_file("mesh/a_2.38.inp");
            
            Assert (dim==2, ExcInternalError());
            grid_in.read_ucd (input_file);
           
            const Point<2> center (0,0);
            static const SphericalManifold<dim> manifold_description;
            triangulation.set_manifold (1, manifold_description);
            
        }
        
        
        for (unsigned int refinement_cycle = 0; refinement_cycle<max_cycle;
             ++refinement_cycle)
        {
            std::cout << "Refinement cycle " << refinement_cycle << std::endl;
            
            if (refinement_cycle == 0)
            {
                std::cout << "For the first cycle, No refinement is needed" << std::endl;
            }
            else
            {
                refine_mesh (refinement_cycle,max_shear_rate);
            }
            
            setup_dofs (refinement_cycle);
            assemble_system ();
            solve ();
            
            BlockVector<double> difference;
            difference = solution ;
            previous_solution =solution ;
            
            int iteration_number=0;
           
            
            do{
                iteration_number +=1;
                
                assemble_system();
                solve();
                difference = solution;
                difference -= previous_solution;
                previous_solution=solution;
                
                std::cout << "   Iteration Number showing : " << iteration_number << "     Difference Norm : " << difference.l2_norm() << std::endl << std::flush;
                
            }while (difference.l2_norm()> pow(10,-9)* dof_handler.n_dofs());
            
            post_processing ();
            output_results (refinement_cycle);
            compute_drag ();
            
            std::cout << "   Refinement..." << std::endl  << std::flush;
            std::cout << std::endl;
            
            std::cout << "   Final Iteration Number: " << iteration_number << "     Difference Norm : " << difference.l2_norm() << std::endl << std::flush;
            std::cout << "   Refinement..." << std::endl  << std::flush;
            std::cout << std::endl;
            

        }
        
    }
    
}

int main ()
{
    
    
    try
    {
        using namespace dealii;
        using namespace MyStokes;
        
        G_u_inflow_n=45.5*pow(10,-3);
        
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
