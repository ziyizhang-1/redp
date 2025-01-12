use super::DataframeSeqExt;
use polars::frame::row::AnyValueBufferTrusted;
use polars::prelude::*;
use polars_arrow::array::PrimitiveArray;
use polars_arrow::bitmap::Bitmap;
use rayon::iter::Either;
use std::borrow::Cow;

impl DataframeSeqExt for DataFrame {
    fn fill_null_seq(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        let col = try_apply_columns(self, &|s| s.fill_null(strategy))?;

        Ok(unsafe { DataFrame::new_no_checks(col) })
    }

    fn drop_nulls_seq<S: AsRef<str>>(&self, subset: Option<&[S]>) -> PolarsResult<Self> {
        let selected_series;

        let mut iter = match subset {
            Some(cols) => {
                selected_series = self.select_series(cols)?;
                selected_series.iter()
            }
            None => self.get_columns().iter(),
        };

        // fast path for no nulls in df
        if iter.clone().all(|s| !s.has_validity()) {
            return Ok(self.clone());
        }

        let mask = iter
            .next()
            .ok_or_else(|| polars_err!(NoData: "no data to drop nulls from"))?;
        let mut mask = mask.is_not_null();

        for s in iter {
            mask = mask & s.is_not_null();
        }
        self._filter_seq(&mask)
    }

    fn transpose_from_dtype(
        &self,
        dtype: &DataType,
        keep_names_as: Option<&str>,
        names_out: &[String],
    ) -> PolarsResult<DataFrame> {
        let new_width = self.height();
        let new_height = self.width();
        // Allocate space for the transposed columns, putting the "row names" first if needed
        let mut cols_t = match keep_names_as {
            None => Vec::<Series>::with_capacity(new_width),
            Some(name) => {
                let mut tmp = Vec::<Series>::with_capacity(new_width + 1);
                tmp.push(StringChunked::new(name, self.get_column_names()).into());
                tmp
            }
        };

        let cols = self.get_columns();
        match dtype {
            DataType::Int32 => numeric_transpose::<Int32Type>(cols, names_out, &mut cols_t),
            DataType::Int64 => numeric_transpose::<Int64Type>(cols, names_out, &mut cols_t),
            DataType::UInt32 => numeric_transpose::<UInt32Type>(cols, names_out, &mut cols_t),
            DataType::UInt64 => numeric_transpose::<UInt64Type>(cols, names_out, &mut cols_t),
            DataType::Float32 => numeric_transpose::<Float32Type>(cols, names_out, &mut cols_t),
            DataType::Float64 => numeric_transpose::<Float64Type>(cols, names_out, &mut cols_t),
            _ => {
                let phys_dtype = dtype.to_physical();
                let mut buffers = (0..new_width)
                    .map(|_| {
                        let buf: AnyValueBufferTrusted = (&phys_dtype, new_height).into();
                        buf
                    })
                    .collect::<Vec<_>>();

                let columns = self
                    .get_columns()
                    .iter()
                    // first cast to supertype before casting to physical to ensure units are correct
                    .map(|s| s.cast(dtype).unwrap().cast(&phys_dtype).unwrap())
                    .collect::<Vec<_>>();

                // this is very expensive. A lot of cache misses here.
                // This is the part that is performance critical.
                for s in columns {
                    polars_ensure!(s.dtype() == &phys_dtype, ComputeError: "cannot transpose with supertype: {}", dtype);
                    s.iter().zip(buffers.iter_mut()).for_each(|(av, buf)| {
                        // SAFETY: we checked the type and we borrow
                        unsafe {
                            buf.add_unchecked_borrowed_physical(&av);
                        }
                    });
                }
                cols_t.extend(buffers.into_iter().zip(names_out).map(|(buf, name)| {
                    // SAFETY: we are casting back to the supertype
                    let mut s = unsafe { buf.into_series().cast_unchecked(dtype).unwrap() };
                    s.rename(name);
                    s
                }));
            }
        };
        Ok(unsafe { DataFrame::new_no_checks(cols_t) })
    }

    /// Transpose a DataFrame. This is a very expensive operation.
    fn transpose_seq(
        &mut self,
        keep_names_as: Option<&str>,
        new_col_names: Option<Either<String, Vec<String>>>,
    ) -> PolarsResult<DataFrame> {
        // We must iterate columns as [`AnyValue`], so we must be contiguous.
        self.as_single_chunk();

        let mut df = Cow::Borrowed(self); // Can't use self because we might drop a name column
        let names_out = match new_col_names {
            None => (0..self.height()).map(|i| format!("column_{i}")).collect(),
            Some(cn) => match cn {
                Either::Left(name) => {
                    let new_names = self.column(&name).and_then(|x| x.str())?;
                    polars_ensure!(new_names.null_count() == 0, ComputeError: "Column with new names can't have null values");
                    df = Cow::Owned(self.drop(&name)?);
                    new_names
                        .into_no_null_iter()
                        .map(|s| s.to_owned())
                        .collect()
                }
                Either::Right(names) => {
                    polars_ensure!(names.len() == self.height(), ShapeMismatch: "Length of new column names must be the same as the row count");
                    names
                }
            },
        };
        if let Some(cn) = keep_names_as {
            // Check that the column name we're using for the original column names is unique before
            // wasting time transposing
            polars_ensure!(names_out.iter().all(|a| a.as_str() != cn), Duplicate: "{} is already in output column names", cn)
        }
        polars_ensure!(
            df.height() != 0 && df.width() != 0,
            NoData: "unable to transpose an empty DataFrame"
        );
        let dtype = df.get_supertype().unwrap()?;

        df.transpose_from_dtype(&dtype, keep_names_as, &names_out)
    }
}

#[inline]
unsafe fn add_value<T: NumericNative>(
    values_buf_ptr: usize,
    col_idx: usize,
    row_idx: usize,
    value: T,
) {
    let column = (*(values_buf_ptr as *mut Vec<Vec<T>>)).get_unchecked_mut(col_idx);
    let el_ptr = column.as_mut_ptr();
    *el_ptr.add(row_idx) = value;
}

// This just fills a pre-allocated mutable series vector, which may have a name column.
// Nothing is returned and the actual DataFrame is constructed above.
pub(super) fn numeric_transpose<T>(cols: &[Series], names_out: &[String], cols_t: &mut Vec<Series>)
where
    T: PolarsNumericType,
    //S: AsRef<str>,
    ChunkedArray<T>: IntoSeries,
{
    let new_width = cols[0].len();
    let new_height = cols.len();

    let has_nulls = cols.iter().any(|s| s.null_count() > 0);

    let mut values_buf: Vec<Vec<T::Native>> = (0..new_width)
        .map(|_| Vec::with_capacity(new_height))
        .collect();
    let mut validity_buf: Vec<_> = if has_nulls {
        // we first use bools instead of bits, because we can access these in parallel without aliasing
        (0..new_width).map(|_| vec![true; new_height]).collect()
    } else {
        (0..new_width).map(|_| vec![]).collect()
    };

    // work with *mut pointers because we it is UB write to &refs.
    let values_buf_ptr = &mut values_buf as *mut Vec<Vec<T::Native>> as usize;
    let validity_buf_ptr = &mut validity_buf as *mut Vec<Vec<bool>> as usize;

    cols.iter().enumerate().for_each(|(row_idx, s)| {
        let s = s.cast(&T::get_dtype()).unwrap();
        let ca = s.unpack::<T>().unwrap();

        // SAFETY:
        // we access in parallel, but every access is unique, so we don't break aliasing rules
        // we also ensured we allocated enough memory, so we never reallocate and thus
        // the pointers remain valid.
        if has_nulls {
            for (col_idx, opt_v) in ca.iter().enumerate() {
                match opt_v {
                    None => unsafe {
                        let column =
                            (*(validity_buf_ptr as *mut Vec<Vec<bool>>)).get_unchecked_mut(col_idx);
                        let el_ptr = column.as_mut_ptr();
                        *el_ptr.add(row_idx) = false;
                        // we must initialize this memory otherwise downstream code
                        // might access uninitialized memory when the masked out values
                        // are changed.
                        add_value(values_buf_ptr, col_idx, row_idx, T::Native::default());
                    },
                    Some(v) => unsafe {
                        add_value(values_buf_ptr, col_idx, row_idx, v);
                    },
                }
            }
        } else {
            for (col_idx, v) in ca.into_no_null_iter().enumerate() {
                unsafe {
                    let column =
                        (*(values_buf_ptr as *mut Vec<Vec<T::Native>>)).get_unchecked_mut(col_idx);
                    let el_ptr = column.as_mut_ptr();
                    *el_ptr.add(row_idx) = v;
                }
            }
        }
    });

    let iter = values_buf.into_iter().zip(validity_buf).zip(names_out).map(
        |((mut values, validity), name)| {
            // SAFETY:
            // all values are written we can now set len
            unsafe {
                values.set_len(new_height);
            }

            let validity = if has_nulls {
                let validity = Bitmap::from_trusted_len_iter(validity.iter().copied());
                if validity.unset_bits() > 0 {
                    Some(validity)
                } else {
                    None
                }
            } else {
                None
            };

            let arr = PrimitiveArray::<T::Native>::new(
                T::get_dtype().to_arrow(true),
                values.into(),
                validity,
            );
            ChunkedArray::with_chunk(name.as_str(), arr).into_series()
        },
    );
    cols_t.extend(iter);
}

// Reduce monomorphization.
fn try_apply_columns(
    df: &DataFrame,
    func: &(dyn Fn(&Series) -> PolarsResult<Series> + Send + Sync),
) -> PolarsResult<Vec<Series>> {
    df.get_columns().iter().map(func).collect()
}
