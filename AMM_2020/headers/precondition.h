// This part has its own tolerance.
// Because preconditioner includes inverse matrix, and to acquire inverse matrix, you should define tolerance


using namespace dealii;

template <int dim>
struct InnerPreconditioner;

template <>
struct InnerPreconditioner<2>
{
    typedef SparseDirectUMFPACK type;
};

template <>
struct InnerPreconditioner<3>
{
    typedef SparseILU<double> type;
};



// ==========================================================
//Precondition of System Matrix for  Flow solver 
// Matrix Solver, for precondition
template <class Matrix, class Preconditioner>
class InverseMatrix : public Subscriptor
{
public:
    InverseMatrix (const Matrix         &m,
                   const Preconditioner &preconditioner);
    
    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;
    
private:
    const SmartPointer<const Matrix> matrix;
    const SmartPointer<const Preconditioner> preconditioner;
};


template <class Matrix, class Preconditioner>
InverseMatrix<Matrix,Preconditioner>::InverseMatrix (const Matrix &m,
                                                     const Preconditioner &preconditioner)
:
matrix (&m),
preconditioner (&preconditioner)
{}


template <class Matrix, class Preconditioner>
void InverseMatrix<Matrix,Preconditioner>::vmult (Vector<double>       &dst,
                                                  const Vector<double> &src) const
{
    SolverControl solver_control (10 * src.size(), pow(10,-7)*src.l2_norm());
    SolverCG<>    cg (solver_control);
    
    dst = 0;
    
    cg.solve (*matrix, dst, src, *preconditioner);
}


template <class Preconditioner>
class SchurComplement : public Subscriptor
{
public:
    SchurComplement (const BlockSparseMatrix<double> &system_matrix,
                     const InverseMatrix<SparseMatrix<double>, Preconditioner> &A_inverse);
    
    void vmult (Vector<double>       &dst,
                const Vector<double> &src) const;
    
private:
    const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
    const SmartPointer<const InverseMatrix<SparseMatrix<double>, Preconditioner> > A_inverse;
    
    mutable Vector<double> tmp1, tmp2;
};



template <class Preconditioner>
SchurComplement<Preconditioner>::
SchurComplement (const BlockSparseMatrix<double> &system_matrix,
                 const InverseMatrix<SparseMatrix<double>,Preconditioner> &A_inverse)
:
system_matrix (&system_matrix),
A_inverse (&A_inverse),
tmp1 (system_matrix.block(0,0).m()),
tmp2 (system_matrix.block(0,0).m())
{}


template <class Preconditioner>
void SchurComplement<Preconditioner>::vmult (Vector<double>       &dst,
                                             const Vector<double> &src) const
{
    system_matrix->block(0,1).vmult (tmp1, src);
    A_inverse->vmult (tmp2, tmp1);
    system_matrix->block(1,0).vmult (dst, tmp2);
}

