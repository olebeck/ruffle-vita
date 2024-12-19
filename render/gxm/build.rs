use std::path::Path;
use std::process::Command;

static SHADER_COMPILER: &str = "./bin/psp2cgc.exe";

fn compile_shader(source: &str, shader_type: &str) {
    let source_path = Path::new(source);
    let shader_name = source_path.file_stem().unwrap_or_default();
    let dest = Path::new("shaders/compiled").join(shader_name).with_extension("gxp");

    let status = Command::new(SHADER_COMPILER)
        .arg(source)
        .args(["-Wperf", "-cache", "-cachedir", "shaders/cache", "-profile", shader_type])
        .args(["-o", dest.to_str().unwrap()])
        .status()
        .expect("Failed to execute shader compiler");

    if !status.success() {
        panic!("Shader compilation failed for source: {}", source);
    }

    println!("cargo:rerun-if-changed={}", source);
}

fn main() {
    if !Path::new(SHADER_COMPILER).exists() {
        panic!("Shader compiler not found at path: {}", SHADER_COMPILER);
    }

    std::fs::create_dir_all("shaders/compiled").unwrap();
    std::fs::create_dir_all("shaders/cache").unwrap();

    compile_shader("shaders/bitmap_f.cg", "sce_fp_psp2");
    compile_shader("shaders/color_f.cg", "sce_fp_psp2");
    compile_shader("shaders/color_v.cg", "sce_vp_psp2");
    compile_shader("shaders/gradient_f.cg", "sce_fp_psp2");
    compile_shader("shaders/texture_v.cg", "sce_vp_psp2");
    compile_shader("shaders/clear_f.cg", "sce_fp_psp2");
    compile_shader("shaders/clear_v.cg", "sce_vp_psp2");
}