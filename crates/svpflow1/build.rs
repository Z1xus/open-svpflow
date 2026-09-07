fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libsvpflow1_vs.dylib");
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        println!("cargo:rustc-cdylib-link-arg=/DEF:{dir}\\svpflow1_vs.def");
        println!("cargo:rustc-cdylib-link-arg=/Brepro");
    }
}
