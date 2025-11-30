#![allow(dead_code)]
use std::cmp;
use std::fmt;

struct Matrix {
    data: Vec<f32>,
    n_rows: usize,
    n_cols: usize,
}

type Shape = (usize, usize);

impl Matrix {
    fn new(data: Vec<f32>, shape: Shape) -> Self {
        let (n_rows, n_cols) = shape;
        Self {
            data,
            n_rows,
            n_cols,
        }
    }

    fn with_value(val: f32, shape: Shape) -> Self {
        let (n_rows, n_cols) = shape;
        Self {
            data: vec![val; n_rows * n_cols],
            n_rows,
            n_cols,
        }
    }

    fn with_range(start: usize, end: usize, shape: Shape) -> Self {
        let (n_rows, n_cols) = shape;
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
    fn det(&self) -> Result<f32, ()> {
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
                    let submatrix = Matrix::new(data, (n - 1, n - 1));
                    acc += (-1.0_f32).powi(col as i32) * self.get(0, col)? * submatrix.det()?;
                }
                Ok(acc)
            }
        }
    }

    /// Performs LU decomposition using Doolittle's method (without pivoting).
    /// Complexity: O(n^3)
    pub fn lu(&self) -> Result<(Matrix, Matrix), ()> {
        if self.n_rows != self.n_cols {
            eprintln!("LU decomposition requires a square matrix");
        }

        let n = self.n_rows;
        let mut lower = Matrix::with_value(0.0, (n, n));
        let mut upper = Matrix::with_value(0.0, (n, n));

        for i in 0..n {
            // Upper triangular U
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..i {
                    sum += lower.get(i, k)? * upper.get(k, j)?;
                }
                upper.set(i, j, self.get(i, j)? - sum)?;
            }

            // Lower triangular L
            for j in i..n {
                if i == j {
                    // Set diagonal to 1
                    lower.set(i, i, 1.0)?;
                } else {
                    let mut sum = 0.0;
                    for k in 0..i {
                        sum += lower.get(j, k)? * upper.get(k, i)?;
                    }

                    let denom = upper.get(i, i)?;
                    if denom.abs() < 1e-6 {
                        eprintln!("Singularity (zero-pivot)");
                    }

                    lower.set(j, i, (self.get(j, i)? - sum) / denom)?;
                }
            }
        }

        Ok((lower, upper))
    }

    fn matmul(&self, other: &Self) -> Result<Self, ()> {
        let (m1, n1) = self.shape();
        let (_, n2) = other.shape();

        let mut mat = Matrix::with_value(0.0, (m1, n2));
        for i in 0..m1 {
            for k in 0..n1 {
                let val_ik = self.get(i, k)?;
                for j in 0..n2 {
                    let val_kj = other.get(k, j)?;
                    let val_ij = mat.get(i, j)?;
                    mat.set(i, j, val_ij + val_ik * val_kj)?;
                }
            }
        }

        Ok(mat)
    }

    fn transpose(&self) -> Result<Self, ()> {
        let (m, n) = self.shape();
        let mut mat = Matrix::new(self.data.clone(), (n, m));
        for i in 0..m {
            for j in 0..n {
                mat.set(j, i, self.get(i, j)?)?;
            }
        }
        Ok(mat)
    }

    fn trace(&self) -> Result<f32, ()> {
        let mut res = 0.0;
        let mut idx = 0;
        while idx < self.n_rows && idx < self.n_cols {
            res += self.get(idx, idx)?;
            idx += 1;
        }
        Ok(res)
    }

    fn gram(&self) -> Result<Self, ()> {
        self.transpose()?.matmul(self)
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
        let val = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let res = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        assert_eq!(res, val);
    }

    #[test]
    fn with_value() {
        let val = Matrix::with_value(4.0, (2, 3));
        let res = Matrix::new(vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0], (2, 3));
        assert_eq!(res, val);
    }

    #[test]
    fn shape() {
        let val = Matrix::with_value(4.0, (2, 3)).shape();
        let res = (2, 3);
        assert_eq!(res, val);
    }

    #[test]
    fn size() {
        let val = Matrix::with_value(4.0, (2, 3)).size();
        let res = 2 * 3;
        assert_eq!(res, val);
    }

    #[test]
    fn bytes() {
        let val = Matrix::with_value(4.0, (2, 3)).bytes();
        let res = 2 * 3 * 4;
        assert_eq!(res, val);
    }

    #[test]
    fn get() -> Result<(), ()> {
        let val = Matrix::with_range(3, 9, (2, 3)).get(1, 1)?;
        let res = 7.0;
        assert_eq!(res, val);
        Ok(())
    }

    #[test]
    fn set() -> Result<(), ()> {
        let mut val = Matrix::with_range(3, 9, (2, 3));
        val.set(1, 1, 8.0)?;
        let res = 8.0;
        assert_eq!(res, val.get(1, 1)?);
        Ok(())
    }

    #[test]
    fn det() -> Result<(), ()> {
        let mat = Matrix::new(
            vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0],
            (3, 3),
        )
        .det()?;
        let res = -78.0;
        assert_eq!(res, mat);
        Ok(())
    }

    #[test]
    fn lu() -> Result<(), ()> {
        let (lower, upper) = Matrix::new(
            vec![1.0, 2.0, -4.0, 2.0, 12.0, 6.0, 7.0, -12.0, 8.0],
            (3, 3),
        )
        .lu()?;
        let res_lower = Matrix::new(vec![1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 7.0, -3.25, 1.0], (3, 3));
        let res_upper = Matrix::new(vec![1.0, 2.0, -4.0, 0.0, 8.0, 14.0, 0.0, 0.0, 81.5], (3, 3));
        assert_eq!(res_lower, lower);
        assert_eq!(res_upper, upper);
        Ok(())
    }

    #[test]
    fn matmul() -> Result<(), ()> {
        let mat1 = Matrix::new((0..30).map(|x| x as f32).collect(), (2, 15));
        let mat2 = Matrix::new((0..30).map(|x| x as f32).collect(), (15, 2));
        let mat3 = mat1.matmul(&mat2)?;
        let res = Matrix::new(vec![2030.0, 2135.0, 5180.0, 5510.0], (2, 2));
        assert_eq!(res, mat3);
        Ok(())
    }

    #[test]
    fn transpose() -> Result<(), ()> {
        let mat = Matrix::new((0..6).map(|x| x as f32).collect(), (2, 3)).transpose()?;
        let res = Matrix::new(vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0], (3, 2));
        assert_eq!(res, mat);
        Ok(())
    }

    #[test]
    fn trace() -> Result<(), ()> {
        let mat = Matrix::new((0..9).map(|x| x as f32).collect(), (3, 3)).transpose()?;
        let res = mat.trace()?;
        assert_eq!(res, 12.0);
        Ok(())
    }

    #[test]
    fn gram() -> Result<(), ()> {
        let _mat = Matrix::new((0..9).map(|x| x as f32).collect(), (3, 3));
        let mat = _mat.transpose()?.matmul(&_mat)?;
        let res = _mat.gram()?;
        eprintln!("{}", mat);
        assert_eq!(res, mat);
        Ok(())
    }

    #[test]
    fn to_string() {
        let val = Matrix::new((1..10).map(|x| x as f32).collect(), (3, 3));
        let res = "[[1, 2, 3]\n [4, 5, 6]\n [7, 8, 9]]\n";
        assert_eq!(res, val.to_string());
    }
}
