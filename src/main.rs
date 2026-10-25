use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};

// Asignadores de memoria de ultra-baja fragmentación para producción
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(target_env = "msvc")]
use mimalloc::MiMalloc;

#[cfg(target_env = "msvc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use remtk_prediction_server::api::{
    build_embed_router, build_predict_router, build_qdrant_router, AppState,
};
use remtk_prediction_server::core::embedding::EmbeddingService;
use remtk_prediction_server::core::inference::InferenceEngine;
use remtk_prediction_server::core::prediction::PredictionService;
use remtk_prediction_server::shared::config::AppConfig;
use remtk_prediction_server::shared::logger::init_logger;
use remtk_prediction_server::support::qdrant::QdrantService;
use remtk_prediction_server::support::registry::TenantCache;
use remtk_prediction_server::support::spatial::SpatialService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 0. Cargar variables de entorno desde .env si existe
    dotenvy::dotenv().ok();

    // 1. Inicializar observabilidad estructurada dual (consola + archivo rotativo diario en logs/)
    let log_cfg = init_logger();

    info!("============================================================");
    info!("🚀 Iniciando remtk-prediction (Monolito Modular en Rust Nativo)");
    info!("============================================================");
    info!(
        "boot logger: dir={} file_enabled={} json_format={} filter=\"{}\"",
        log_cfg.logs_dir.display(),
        log_cfg.file_logging_enabled,
        log_cfg.json_format,
        log_cfg.env_filter
    );

    // 2. Cargar configuración desde variables de entorno
    let config = Arc::new(AppConfig::from_env());
    info!(
        "boot config: predict={} embed={} qdrant={}",
        config.port_predict, config.port_embed, config.port_qdrant
    );
    info!(
        "boot models: path={} size={} intra_threads={} inter_threads={}",
        config.models_path.display(),
        config.model_size,
        config.intra_op_threads,
        config.inter_op_threads
    );

    // 3. Inicializar el motor de inferencia matemática puro (core/inference)
    info!("boot: Cargando sesiones neuronales ONNX Runtime con mmap...");
    let engine = Arc::new(InferenceEngine::new(&config)?);
    info!("boot: Motor ONNX listo. multilingual-e5-small y remtk-juicio-moe en memoria compartida.");

    // 4. Inicializar capas de soporte (support/)
    let cache = Arc::new(TenantCache::new());
    let spatial = Arc::new(SpatialService::new());
    let qdrant = Arc::new(QdrantService::new(cache.clone(), engine.clone()));

    // 5. Inicializar servicios de dominio central (core/)
    let embedding = Arc::new(EmbeddingService::new(engine.clone()));
    let prediction = Arc::new(PredictionService::new(
        engine.clone(),
        cache.clone(),
        config.clone(),
    ));

    // 6. Ensamblar estado de aplicación (api/state.rs)
    let state = AppState::new(
        prediction,
        embedding,
        spatial,
        qdrant,
        cache,
        config.clone(),
    );

    // 7. Construir los routers HTTP independientes
    let app_predict = build_predict_router(state.clone());
    let app_embed = build_embed_router(state.clone());
    let app_qdrant = build_qdrant_router(state.clone());

    let addr_predict: SocketAddr = format!("0.0.0.0:{}", config.port_predict).parse()?;
    let addr_embed: SocketAddr = format!("0.0.0.0:{}", config.port_embed).parse()?;
    let addr_qdrant: SocketAddr = format!("0.0.0.0:{}", config.port_qdrant).parse()?;

    // 8. Bind de los tres listeners TCP
    let listener_predict = tokio::net::TcpListener::bind(addr_predict).await?;
    let listener_embed = tokio::net::TcpListener::bind(addr_embed).await?;
    let listener_qdrant = tokio::net::TcpListener::bind(addr_qdrant).await?;

    info!("→ Predicción  http://localhost:{}", config.port_predict);
    info!("→ Modelos     http://localhost:{}", config.port_embed);
    info!("→ Qdrant      http://localhost:{}", config.port_qdrant);
    info!("⚡ Monolito Modular activo. Presupuesto de memoria objetivo: < 90 MB RSS.");

    // 9. Bucle concurrente unificado con apagado elegante (graceful shutdown)
    tokio::select! {
        res = axum::serve(listener_predict, app_predict).with_graceful_shutdown(shutdown_signal()) => {
            if let Err(e) = res {
                error!("Error en listener de predicción (6776): {}", e);
            }
        }
        res = axum::serve(listener_embed, app_embed).with_graceful_shutdown(shutdown_signal()) => {
            if let Err(e) = res {
                error!("Error en listener de modelos (6777): {}", e);
            }
        }
        res = axum::serve(listener_qdrant, app_qdrant).with_graceful_shutdown(shutdown_signal()) => {
            if let Err(e) = res {
                error!("Error en listener de Qdrant (6775): {}", e);
            }
        }
    }

    info!("Apagado elegante completado. Todos los sockets han sido cerrados.");
    Ok(())
}

/// Manejador de señales de terminación para drenado de conexiones activas.
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Fallo al registrar manejador de señal Ctrl+C");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Fallo al registrar manejador de señal SIGTERM")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Señal SIGINT / Ctrl+C interceptada. Iniciando drenado de conexiones...");
        }
        _ = terminate => {
            info!("Señal SIGTERM interceptada. Iniciando drenado de conexiones...");
        }
    }
}
