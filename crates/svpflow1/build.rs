fn main() {
    if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        println!("cargo:rustc-cdylib-link-arg=/DEF:{dir}\\svpflow1_vs.def");
    }
}
