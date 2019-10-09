//
//  bc_and_rhs.h
//  
//
//  Created by Kim, Jaekwang on 7/23/17.
//
//

#ifndef bc_and_rhs_h
#define bc_and_rhs_h

#include <stdio.h>

using namespace dealii;


/*
//header file tester :
int square (int a);

int square (int a)
{
    return a*a;
}
//end of header file tester
*/


//Global Variable being used in main_function
double G_u_inflow_n;
double G_drag_force_n;


// Dirichlet Constant Uz BC
template <int dim>
class ConstantUz : public Function<dim>
{
public:
    ConstantUz  () : Function<dim>(dim+1) {}
    
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
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));
    
    double u = G_u_inflow_n;  //Read from outside global variable
    
    if (component == 1)
        return u;
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


// RightHandSide Vector Function for GNF SYSTEM ASSEMBLE
// This class was developed to test the code with manufactured solution, in this case, we need non-zero right_hand_side
template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide () : Function<dim>(dim+1) {}
    
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    
};

template <int dim>
void
RightHandSide<dim>::vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const
{
    
    values(0) =  0;
    values(1) =  0;
    values(2) =  0;
}

#endif /* header_h */
