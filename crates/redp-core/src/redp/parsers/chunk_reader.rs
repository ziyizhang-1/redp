use grep::regex::RegexMatcher;
use grep::searcher::{Searcher, Sink, SinkMatch};
use rayon::iter::Either;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io;
use std::path::Path;
use std::str::from_utf8;

type Error = Box<dyn std::error::Error + 'static>;

#[derive(Clone, Debug)]
struct Bytes<F>(pub F)
where
    F: FnMut(u64, &[u8]) -> Result<bool, Error>;

impl<F> Sink for Bytes<F>
where
    F: FnMut(u64, &[u8]) -> Result<bool, Error>,
{
    type Error = Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, Error> {
        let bytes_offset = mat.absolute_byte_offset();
        (self.0)(bytes_offset, mat.bytes())
    }
}

pub fn update_unique_uncore_devices(
    block: &[u8],
    unique_uncore_devices: &mut HashSet<String>,
) -> Result<(), Error> {
    Searcher::new().search_slice(
        RegexMatcher::new(r"^UNC_")?,
        block,
        Bytes(|_, line| {
            let device: &[u8] = line
                .strip_prefix(b"UNC_")
                .unwrap()
                .split(|e| *e == b'_')
                .next()
                .unwrap();
            unique_uncore_devices.insert("UNC_".to_string() + from_utf8(device)?);
            Ok(true)
        }),
    )?;
    Ok(())
}

pub fn get_block_offsets(
    path: Either<impl AsRef<Path>, &[u8]>,
    pattern: &str,
) -> Result<Vec<(usize, usize)>, Error> {
    let mut offsets: Vec<(usize, usize)> = vec![];

    match path {
        Either::Left(v) => {
            let file = OpenOptions::new().read(true).open(v)?;
            Searcher::new().search_file(
                RegexMatcher::new(pattern)?, // r"==========|Version Info:"
                &file,
                Bytes(|offset, line| {
                    offsets.push((offset as usize, offset as usize + line.len()));
                    Ok(true)
                }),
            )?;
        }
        Either::Right(v) => {
            Searcher::new().search_slice(
                RegexMatcher::new(pattern)?, // r"==========|Version Info:"
                v,
                Bytes(|offset, line| {
                    offsets.push((offset as usize, offset as usize + line.len()));
                    Ok(true)
                }),
            )?;
        }
    };

    if offsets.is_empty() {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
    }

    Ok(offsets)
}

#[derive(Clone)]
pub struct Chunk<'a, T> {
    pub inner: &'a [T],
    pub start_idx: usize,
}

pub trait ChunkRead<T> {
    fn read_in_chunks(
        &self,
        offsets: Option<Vec<(usize, usize)>>,
        pattern: Option<&str>,
        chunk_size: Option<u32>,
    ) -> Result<Vec<Chunk<T>>, Error>;
}

impl ChunkRead<u8> for [u8] {
    fn read_in_chunks(
        &self,
        offsets: Option<Vec<(usize, usize)>>,
        pattern: Option<&str>,
        chunk_size: Option<u32>,
    ) -> Result<Vec<Chunk<u8>>, Error> {
        let offsets = match offsets {
            Some(v) => v,
            None => get_block_offsets(Either::<&Path, &[u8]>::Right(self), pattern.unwrap())?,
        };
        let num_blocks = offsets.len();
        let mut chunk_start: usize = offsets[0].1;
        let len = self.len();

        let chunk_size = match chunk_size {
            Some(v) => {
                if v == 0 {
                    num_blocks as u32
                } else {
                    v
                }
            }
            None => {
                if num_blocks < 100 {
                    5_u32
                } else if num_blocks < 500 {
                    10_u32
                } else {
                    10 + 5 * (num_blocks / 500) as u32
                }
            }
        };

        let mut chunk_offsets: Vec<(usize, usize, usize)> = offsets
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 0 && i % chunk_size as usize == 0)
            .map(|(i, (_start, _end))| {
                let start = chunk_start;
                let end = *_start;
                chunk_start = *_end;
                (start, end, 1 + i - chunk_size as usize)
            })
            .collect();

        if chunk_start < len {
            if let Some(v) = chunk_offsets.last() {
                chunk_offsets.push((chunk_start, len, v.2 + chunk_size as usize));
            } else {
                chunk_offsets.push((chunk_start, len, 1));
            }
        }

        let chunks: Vec<Chunk<u8>> = chunk_offsets
            .into_iter()
            .map(move |(left, right, idx)| Chunk {
                inner: &self[left..right],
                start_idx: idx,
            })
            .collect();

        Ok(chunks)
    }
}
