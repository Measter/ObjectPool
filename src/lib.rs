use std::{error::Error, fmt::Display};

mod chunk_manager;
pub mod pool;
pub mod slab;

#[derive(Debug, Clone, Copy)]
pub enum ErrorKind {
    CapacityOverflow,
    AllocatorError,
    TypeTooBig,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("allocation failed")?;
        let msg = match self {
            ErrorKind::CapacityOverflow => " because the computed capacity overflowed",
            ErrorKind::AllocatorError => " because the allocator returned an error",
            ErrorKind::TypeTooBig => " because the type is too big for the slot",
        };
        f.write_str(msg)
    }
}

impl Error for ErrorKind {}
