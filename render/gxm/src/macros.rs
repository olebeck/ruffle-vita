
#[repr(C)] // guarantee 'bytes' comes after '_align'
pub struct AlignedAs<Align, Bytes: ?Sized> {
    pub _align: [Align; 0],
    pub bytes: Bytes,
}

#[macro_export]
macro_rules! include_bytes_align_as {
    ($align_ty:ty, $path:literal) => {
        {  // const block expression to encapsulate the static
            use $crate::macros::AlignedAs;
            
            // this assignment is made possible by CoerceUnsized
            static ALIGNED: &AlignedAs::<$align_ty, [u8]> = &AlignedAs {
                _align: [],
                bytes: *include_bytes!($path),
            };

            &ALIGNED.bytes
        }
    };
}

#[macro_export]
macro_rules! sce_err {
    ($func:expr, $($arg:expr),*) => {{
        //println!(stringify!($func));
        //std::thread::sleep(std::time::Duration::from_millis(10));
        #[allow(unused_unsafe)]
        let err = unsafe { $func($($arg),*) };
        if err < 0 {
            panic!("{} failed with error: 0x{:x}", stringify!($func), err);
        }
        err
    }};
}

#[macro_export]
macro_rules! to_cstring {
    ($s:expr) => {
        std::ffi::CString::new($s).expect("String contains a null byte")
    };
}

#[macro_export]
macro_rules! ALIGN {
    ($x:expr, $a:expr) => {
        ((($x) + (($a) - 1)) & !(($a) - 1))
    };
}
