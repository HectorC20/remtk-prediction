mod e5;
mod model;
mod registry;
mod server;
mod types;

use std::path::Path;
use std::sync::Arc;
use candle_core::Device;
use e5::E5Embedder;
use model::RemtkModel;
use registry::TenantRegistry;
use server::{create_router, AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  REMTK ENGINE RS — Pure Candle Native Prediction Service   ");
    println!("============================================================");

    let dev = Device::Cpu;

    let base_dir = Path::new(r"C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction");
    let model_dir = base_dir.join("models").join("remtk-juicio-jev-moe-3.8b-400tools");
    let e5_dir = base_dir.join("models").join("multilingual-e5-small-onnx");
    let safetensors_path = base_dir.join("models").join("remtk_juicio_jev_moe_3_82B.safetensors");

    println!("[1/3] Loading Multilingual-E5 Embedder from {:?}...", e5_dir);
    let embedder = Arc::new(E5Embedder::load(&e5_dir, &dev)?);
    println!("  -> E5 Embedder initialized!");

    println!("[2/3] Loading TenantRegistry and Catalog Metadata from {:?}...", model_dir);
    let registry = Arc::new(TenantRegistry::load_from_dir(&model_dir, &dev)?);
    println!("  -> Catalog indexed with {} tools!", registry.global_tool_ids.len());

    println!("[3/3] Loading REMTK 3.82B MoE Model from {:?}...", safetensors_path);
    let model = Arc::new(RemtkModel::load_from_safetensors(&safetensors_path, &dev)?);
    println!("  -> REMTK 3.82B MoE Model fully operative!");

    let state = Arc::new(AppState {
        model,
        embedder,
        registry,
        dev,
    });

    println!("\nStarting dual Axum HTTP servers on 0.0.0.0:6776 and 0.0.0.0:6876...");

    let app_6776 = create_router(state.clone());
    let app_6876 = create_router(state.clone());

    let listener_6776 = tokio::net::TcpListener::bind("0.0.0.0:6776").await?;
    let listener_6876 = tokio::net::TcpListener::bind("0.0.0.0:6876").await?;

    println!("REMTK Engine RS listening on http://0.0.0.0:6776 and http://0.0.0.0:6876");

    let server_6776 = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener_6776, app_6776).await {
            eprintln!("Error on 6776: {}", e);
        }
    });

    let server_6876 = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener_6876, app_6876).await {
            eprintln!("Error on 6876: {}", e);
        }
    });

    let _ = tokio::try_join!(server_6776, server_6876)?;

    Ok(())
}
