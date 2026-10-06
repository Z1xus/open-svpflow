fn main() {
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libopen_svpflow.dylib");
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,@loader_path");
        }
        Ok("linux") => println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN"),
        _ => {}
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-cdylib-link-arg=/Brepro");
    }
}
