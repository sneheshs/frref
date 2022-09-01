import numpy as np
from scipy.sparse import isspmatrix  # ,csr_matrix


def frref(A, TOL=None, TYPE=''):
    '''
    %FRREF   Fast reduced row echelon form.
    %   R = FRREF(A) produces the reduced row echelon form of A.
    %   [R,jb] = FRREF(A,TOL) uses the given tolerance in the rank tests.
    %   [R,jb] = FRREF(A,TOL,TYPE) forces frref calculation using the algorithm
    %   for full (TYPE='f') or sparse (TYPE='s') matrices.
    %
    %
    %   Description:
    %   For full matrices, the algorithm is based on the vectorization of MATLAB's
    %   RREF function. A typical speed-up range is about 2-4 times of
    %   the MATLAB's RREF function. However, the actual speed-up depends on the
    %   size of A. The speed-up is quite considerable if the number of columns in
    %   A is considerably larger than the number of its rows or when A is not dense.
    %
    %   For sparse matrices, the algorithm ignores the TOL value and uses sparse
    %   QR to compute the rref form, improving the speed by a few orders of
    %   magnitude.
    %
    %   Authors: Armin Ataei-Esfahani (2008)
    %            Ashish Myles (2012)
    #            Snehesh Shrestha (2020)
    %
    %   Revisions:
    %   25-Sep-2008   Created Function
    %   21-Nov-2012   Added faster algorithm for sparse matrices
    #   30-June-2020  Ported to python. TODO: Only do_full implemented. The remaining of the function. See frref_orig below.
    '''

    m = np.shape(A)[0]
    n = np.shape(A)[1]

    # Process Arguments
    # ----------------------------------------------------------
    # TYPE -- Sparce (s) or non-Sparce (Full, f)
    if TYPE == '':   # set TYPE if sparse or not
        if isspmatrix(A):
            TYPE = 's'
        else:
            TYPE = 'f'
    else:   # Set type
        if not type(TYPE) is str or len(TYPE) > 1:  # Check valid type
            print('Unknown matrix TYPE! Use "f" for full and "s" for sparse matrices.')
            exit()

        TYPE = TYPE.lower()
        if not TYPE == 'f' and not TYPE == 's':
            print(
                'Unknown matrix TYPE! Use ''f'' for full and ''s'' for sparse matrices.')
            exit()

    if TYPE=='f':
        # TOLERENCE
        # % Compute the default tolerance if none was provided.
        if TOL is None:
            # Prior commit had TOL default to 1e-6
            # TOL = max(m,n)*eps(class(A))*norm(A,'inf')
            TOL = max(m, n)*np.finfo(A.dtype).eps*np.linalg.norm(A, np.inf)

    # Non-Sparse
    # ----------------------------------------------------------
    if not isspmatrix(A) or TYPE == 'f':
        # % Loop over the entire matrix.
        i = 0
        j = 0
        jb = []

        while (i < m) and (j < n):
            # % Find value (p) and index (k) of largest element in the remainder of column j.
            abscol = np.array(np.abs(A[i:m, j]))
            p = np.max(abscol)
            k = np.argmax(abscol, axis=0)
            if np.ndim(k) > 1:
                k = k[0]
            else:
                k = int(k)

            k = k+i  # -1 #python zero index, not needed

            if p <= TOL:
                # % The column is negligible, zero it out.
                A[i:m, j] = 0  # %(faster for sparse) %zeros(m-i+1,1);
                j += 1
            else:
                # % Remember column index
                jb.append(j)

                # % Swap i-th and k-th rows.
                A = np.array(A)
                A[[i, k], j:n] = A[[k, i], j:n]

                # % Divide the pivot row by the pivot element.
                Ai = np.nan_to_num(A[i, j:n] / A[i, j])
                Ai = np.matrix(Ai).T.T

                # % Subtract multiples of the pivot row from all the other rows.
                A[:, j:n] = A[:, j:n] - np.dot(A[:, [j]], Ai)
                A[i, j:n] = Ai
                i += 1
                j += 1

        return A, jb

    # Sparse
    # ----------------------------------------------------------
    else:
        A = np.array(A.toarray())
        return frref(A, TYPE='f')

        # # TODO: QR-decomposition of a Sparse matrix is not so simple in Python -- still need to figure out a solution
        # # % Non-pivoted Q-less QR decomposition computed by Matlab actually
        # # % produces the right structure (similar to rref) to identify independent
        # # % columns.
        # R = numpy.linalg.qr(A)

        # # % i_dep = pivot columns = dependent variables
        # # %       = left-most non-zero column (if any) in each row
        # # % indep_rows (binary vector) = non-zero rows of R
        # [indep_rows, i_dep] = np.max(R ~= 0, [], 2)     # TODO
        # indep_rows = full[indep_rows]; # % probably more efficient
        # i_dep = i_dep[indep_rows]
        # i_indep = setdiff[1:n, i_dep]

        # # % solve R(indep_rows, i_dep) x = R(indep_rows, i_indep)
        # # %   to eliminate all the i_dep columns
        # # %   (i.e. we want F(indep_rows, i_dep) = Identity)
        # F = sparse([],[],[], m, n)
        # F[indep_rows, i_indep] = R[indep_rows, i_dep] \ R[indep_rows, i_indep]
        # F[indep_rows, i_dep] = speye(length(i_dep))

        # # % result
        # A = F
        # jb = i_dep

        # return A, jb
