use std::cmp;
use std::fmt;

struct Matrix {
    data: Vec<f32>,
    n_rows: usize,
    n_cols: usize,
}

type Shape = (usize, usize);

impl Matrix {
    fn new(data: Vec<f32>, n_rows: usize, n_cols: usize) -> Self {
        Self {
            data,
            n_rows,
            n_cols,
        }
    }

    fn with_value(val: f32, n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: vec![val; n_rows * n_cols],
            n_rows,
            n_cols,
        }
    }

    fn with_range(start: usize, end: usize, n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: (start..end).map(|x| x as f32).collect(),
            n_rows,
            n_cols,
        }
    }

    fn shape(&self) -> Shape {
        (self.n_rows, self.n_cols)
    }

    fn size(&self) -> usize {
        self.n_rows * self.n_cols
    }

    fn bytes(&self) -> usize {
        4 * self.n_rows * self.n_cols
    }

    fn get(&self, row: usize, col: usize) -> Result<f32, ()> {
        if row >= self.n_rows || col >= self.n_cols {
            eprintln!("Index out of bounds");
        }
        let idx = self.n_cols * row + col;
        Ok(self.data[idx])
    }

    fn set(&mut self, row: usize, col: usize, val: f32) -> Result<(), ()> {
        if row >= self.n_rows || col >= self.n_cols {
            eprintln!("Index out of bounds");
        }
        let idx = self.n_cols * row + col;
        self.data[idx] = val;
        Ok(())
    }

    /// Computes matrix via Laplace (cofactor) expansion
    /// Complexity: O(n!)
    ///
    /// NOTE: Implement LU decomposition, which has the computational time complexity of O(n^3).
    fn det_laplace(&self) -> Result<f32, ()> {
        if self.n_rows != self.n_cols {
            eprintln!("Determinant is not defined for a non-square matrix");
        }

        match self.shape() {
            (1, 1) => Ok(self.data[0]),
            (2, 2) => Ok(self.get(0, 0)? * self.get(1, 1)? - self.get(0, 1)? * self.get(1, 0)?),
            (n, _) => {
                // We perform Laplace (cofactor) expansion to compute the determinant of a matrix
                // with arbitrary dimensions.
                //
                // Formula: det(A) = Σ (−1)^(i+j) * a_{i,j} * M_{i,j}
                //
                // where a_{i,j} * det(A_{i,j}) is the cofactor, and M_{i,j} = det(A_{i,j}).
                //
                // The expansion can be done across any row i. Here, we expand along row i = 0.
                let mut acc = 0.0;
                for col in 0..n {
                    let mut data: Vec<f32> = Vec::with_capacity((n - 1) * (n - 1));
                    for i in 1..n {
                        for j in 0..n {
                            if j == col {
                                continue;
                            }
                            data.push(self.get(i, j)?);
                        }
                    }
                    let submatrix = Matrix::new(data, n - 1, n - 1);
                    acc += (-1.0_f32).powi(col as i32)
                        * self.get(0, col)?
                        * submatrix.det_laplace()?;
                }
                Ok(acc)
            }
        }
    }

    fn matmul(&self, other: Self) -> Result<Self, ()> {
        let (m1, n1) = self.shape();
        let (m2, n2) = other.shape();

        if n1 != m2 {
            eprintln!("ERROR: Cannot multiply {}x{} with {}x{}", m1, n1, m2, n2);
        }

        let mut m = Matrix::with_value(0.0, m1, n2);
        for i in 0..m1 {
            for k in 0..n1 {
                let val_ik = self.get(i, k)?;
                for j in 0..n2 {
                    let val_kj = other.get(k, j)?;
                    let val_ij = m.get(i, j)?;
                    m.set(i, j, val_ij + val_ik * val_kj)?;
                }
            }
        }

        Ok(m)
    }
}

impl cmp::PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        self.n_rows == other.n_rows && self.n_cols == other.n_cols && self.data == other.data
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        let mut res = String::new();
        res.push('[');
        for i in 0..self.n_rows {
            if i > 0 {
                res.push(' ');
            }
            res.push('[');
            for j in 0..self.n_cols {
                res.push_str(&self.get(i, j).unwrap().to_string());
                if j < self.n_cols - 1 {
                    res.push_str(", ");
                }
            }
            if i == self.n_rows - 1 {
                res.push_str("]]\n");
            } else {
                res.push_str("]\n");
            }
        }
        write!(f, "{}", res)
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn init() {
        let val = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let res = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(res, val);
    }

    #[test]
    fn with_value() {
        let val = Matrix::with_value(4.0, 2, 3);
        let res = Matrix::new(vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0], 2, 3);
        assert_eq!(res, val);
    }

    #[test]
    fn shape() {
        let val = Matrix::with_value(4.0, 2, 3).shape();
        let res = (2, 3);
        assert_eq!(res, val);
    }

    #[test]
    fn size() {
        let val = Matrix::with_value(4.0, 2, 3).size();
        let res = 2 * 3;
        assert_eq!(res, val);
    }

    #[test]
    fn bytes() {
        let val = Matrix::with_value(4.0, 2, 3).bytes();
        let res = 2 * 3 * 4;
        assert_eq!(res, val);
    }

    #[test]
    fn get() -> Result<(), ()> {
        let val = Matrix::with_range(3, 9, 2, 3).get(1, 1)?;
        let res = 7.0;
        assert_eq!(res, val);
        Ok(())
    }

    #[test]
    fn set() -> Result<(), ()> {
        let mut val = Matrix::with_range(3, 9, 2, 3);
        val.set(1, 1, 8.0)?;
        let res = 8.0;
        assert_eq!(res, val.get(1, 1)?);
        Ok(())
    }

    #[test]
    fn det() -> Result<(), ()> {
        let m = Matrix::new(vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0], 3, 3)
            .det_laplace()?;
        let res = -78.0;
        assert_eq!(res, m);
        Ok(())
    }

    #[test]
    fn matmul() -> Result<(), ()> {
        let m1 = Matrix::new((0..30).map(|x| x as f32).collect(), 2, 15);
        let m2 = Matrix::new((0..30).map(|x| x as f32).collect(), 15, 2);
        let m3 = m1.matmul(m2)?;
        let res = Matrix::new(vec![2030.0, 2135.0, 5180.0, 5510.0], 2, 2);
        assert_eq!(res, m3);
        Ok(())
    }

    #[test]
    fn to_string() {
        let val = Matrix::new((1..10).map(|x| x as f32).collect(), 3, 3);
        let res = "[[1, 2, 3]\n [4, 5, 6]\n [7, 8, 9]]\n";
        assert_eq!(res, val.to_string());
    }
}
