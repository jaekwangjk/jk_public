#ifndef bc_and_rhs_h
#define bc_and_rhs_h
#include <stdio.h>

using namespace dealii;
double U_inflow;

template <int dim>
class ConstantUz : public Function<dim>
{
public:
    ConstantUz  () : Function<dim>(dim+2) {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int component = 0) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
};

template <int dim>
double
ConstantUz<dim>::value (const Point<dim>  &p,
                        const unsigned int component) const
{
    if (component == 1)
        return U_inflow;
    else
        return 0;
}


template <int dim>
void
ConstantUz<dim>::vector_value (const Point<dim> &p,
                               Vector<double>   &values) const
{
    for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = ConstantUz<dim>::value (p, c);
}


template <int dim>
class StructureBoundaryValues : public Function<dim>
{
public:
    StructureBoundaryValues () : Function<dim>(1) {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};


template <int dim>
double
StructureBoundaryValues<dim>::value (const Point<dim> &p,
                                     const unsigned int /*component */) const
{
    return 1;
}

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide () : Function<dim>(dim+2) {}
    
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    
};

template <int dim>
void
RightHandSide<dim>::vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const
{
    values(0) = 0;
    values(1) = 0;
    values(2) = 0;
}


template <int dim>
class RightHandSide_trpt : public Function<dim>
{
public:
    RightHandSide_trpt () : Function<dim>() {}
    
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};


template <int dim>
double
RightHandSide_trpt<dim>::value (const Point<dim>   &p,
                                const unsigned int  component) const
{
    Assert (component == 0, ExcIndexRange (component, 0, 1));
    
    double rhs=0;
    return rhs;
    
    
}

#endif /* header_h */
