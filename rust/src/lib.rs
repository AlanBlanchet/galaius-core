//! The rules every reader of galaius data applies.  Each mirrors one definition of the Python package until installed
//! clients ship these (README).
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod files;
#[cfg(feature = "python")]
mod python;
pub mod roles;
pub mod run_state;
pub mod tokens;
