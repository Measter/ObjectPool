use std::{error::Error, fmt::Display};

pub mod slab;

#[derive(Debug, Clone, Copy)]
pub enum ErrorKind {
    CapacityOverflow,
    AllocatorError,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("allocation failed")?;
        let msg = match self {
            ErrorKind::CapacityOverflow => " because the computed capacity overflowed",
            ErrorKind::AllocatorError => " because the allocator returned an error",
        };
        f.write_str(msg)
    }
}

impl Error for ErrorKind {}
