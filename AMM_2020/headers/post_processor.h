#ifndef post_processor_h
#define post_processor_h
#include <stdio.h>

using namespace dealii;

template <int dim>
class ShearRate : public DataPostprocessorScalar<dim>
{
    
public:
    ShearRate ();
    virtual
    void compute_derived_quantities_vector (const std::vector<Vector<double> >               &uh,
                                            const std::vector<std::vector<Tensor<1, dim> > > &duh,
                                            const std::vector<std::vector<Tensor<2, dim> > > &dduh,
                                            const std::vector<Point<dim> >                   &normals,
                                            const std::vector<Point<dim> >                   &evaluation_points,
                                            std::vector<Vector<double> >                     &computed_quantities) const;
};

template <int dim>
ShearRate<dim>::ShearRate ():DataPostprocessorScalar<dim> ("ShearRate", update_values | update_gradients | update_quadrature_points)

{}

template <int dim>
void ShearRate<dim>::compute_derived_quantities_vector (const std::vector<Vector<double> >                 & uh,
                                                        const std::vector<std::vector<Tensor<1, dim> > >   & duh,
                                                        const std::vector<std::vector<Tensor<2, dim> > >   & /*dduh*/,
                                                        const std::vector<Point<dim> >                     & /*normals*/,
                                                        const std::vector<Point<dim> >                     & evaluation_points,
                                                        std::vector<Vector<double> >                       & computed_quantities) const
{
    for (unsigned int i=0; i<computed_quantities.size(); i++)
        
    { Assert(computed_quantities[i].size() == 1,
             ExcDimensionMismatch (computed_quantities[i].size(), 1));
        Assert(uh[i].size() == 4, ExcDimensionMismatch (uh[i].size(), 4));
        
        Tensor<2,dim> grad_u;
        for (unsigned int d=0; d<dim; ++d)
            grad_u[d] = duh[i][d]; // extract the first derivative of velocity component only (from full solution: origianl FEM system (v,p,strucutre))
        const SymmetricTensor<2,dim> shear_rate_tensor = symmetrize (grad_u);
        if(evaluation_points[i](0)==0) //axis singularity, LHOPITALS' RULE:: ( lim_(r->0) [(u_r)/r] = lim_(r->0) (d(u_r)/dr) = duh[i][1];
        {
            computed_quantities[i](0) = std::sqrt(2*(shear_rate_tensor*shear_rate_tensor +  std::pow(grad_u[0][0],2)) );
        }
        else
        {
            computed_quantities[i](0) = std::sqrt(2* (shear_rate_tensor*shear_rate_tensor + std::pow( (uh[i](0) /evaluation_points[i](0)),2) ));
        }
    }
}

template <int dim>
class Viscosity : public DataPostprocessorScalar<dim>
{
    
public:
    Viscosity ();
    virtual
    void compute_derived_quantities_vector (const std::vector<Vector<double> >               &uh,
                                            const std::vector<std::vector<Tensor<1, dim> > > &duh,
                                            const std::vector<std::vector<Tensor<2, dim> > > &dduh,
                                            const std::vector<Point<dim> >                   &normals,
                                            const std::vector<Point<dim> >                   &evaluation_points,
                                            std::vector<Vector<double> >                     &computed_quantities) const;
};


template <int dim>

Viscosity<dim>::Viscosity ():DataPostprocessorScalar<dim> ("Viscosity", update_values | update_gradients | update_quadrature_points)

{}

template <int dim>
void Viscosity<dim>::compute_derived_quantities_vector (
                                                        const std::vector<Vector<double> >                 & uh,
                                                        const std::vector<std::vector<Tensor<1, dim> > >   & /*duh*/,
                                                        const std::vector<std::vector<Tensor<2, dim> > >   & /*dduh*/,
                                                        const std::vector<Point<dim> >                     & /*normals*/,
                                                        const std::vector<Point<dim> >                     & evaluation_points,
                                                        std::vector<Vector<double> >                       & computed_quantities) const
{
    double eta_str=2.014;
    double eta_0=29.0;
    
    for (unsigned int i=0; i<computed_quantities.size(); i++)
        
    { Assert(computed_quantities[i].size() == 1,
             ExcDimensionMismatch (computed_quantities[i].size(), 1));
        Assert(uh[i].size() == 4, ExcDimensionMismatch (uh[i].size(), 4));
        
        double lambda = uh[i][dim+1];
        computed_quantities[i](0) = eta_0 + eta_str * lambda;
    
    }
}

template <int dim>
class Grad_Lambda : public DataPostprocessorVector<dim>
{
    
public:
    Grad_Lambda ();
    virtual
    void compute_derived_quantities_vector (const std::vector<Vector<double> >               &uh,
                                            const std::vector<std::vector<Tensor<1, dim> > > &duh,
                                            const std::vector<std::vector<Tensor<2, dim> > > &dduh,
                                            const std::vector<Point<dim> >                   &normals,
                                            const std::vector<Point<dim> >                   &evaluation_points,
                                            std::vector<Vector<double> >                     &computed_quantities) const;
};


template <int dim>

Grad_Lambda<dim>::Grad_Lambda ():DataPostprocessorVector<dim> ("GRAD_LAM", update_values | update_gradients | update_quadrature_points)

{}

template <int dim>
void Grad_Lambda<dim>::compute_derived_quantities_vector (
                                                        const std::vector<Vector<double> >                 & uh,
                                                        const std::vector<std::vector<Tensor<1, dim> > >   & duh,
                                                        const std::vector<std::vector<Tensor<2, dim> > >   & /*dduh*/,
                                                        const std::vector<Point<dim> >                     & normals,
                                                        const std::vector<Point<dim> >                     & evaluation_points,
                                                        std::vector<Vector<double> >                       & computed_quantities) const
{
    for (unsigned int i=0; i<computed_quantities.size(); i++)
        
    {
        Assert(computed_quantities[i].size() == 2,
               ExcDimensionMismatch (computed_quantities[i].size(), 2));
        Assert(uh[i].size() == 4, ExcDimensionMismatch (uh[i].size(), 4));
        
        Tensor<1,dim> grad_structure;
        grad_structure = duh[i][dim+1];
        computed_quantities[i](0) = grad_structure[0];
        computed_quantities[i](1) = grad_structure[1];
       
    }
}


#endif /* header_h */
