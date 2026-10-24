pub mod embed_server;
pub mod predict_server;
pub mod qdrant_server;
pub mod state;

use std::time::Duration;
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

pub use state::AppState;

fn apply_middleware(router: Router) -> Router {
    router
        .layer(TimeoutLayer::new(Duration::from_secs(15)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

/// Construye el router para el Puerto 6776 (Predicción y Juicio MoE).
pub fn build_predict_router(state: AppState) -> Router {
    apply_middleware(predict_server::create_router(state))
}

/// Construye el router para el Puerto 6777 (Embeddings y Modelos).
pub fn build_embed_router(state: AppState) -> Router {
    apply_middleware(embed_server::create_router(state))
}

/// Construye el router para el Puerto 6775 (Qdrant Proxy y Sincronización).
pub fn build_qdrant_router(state: AppState) -> Router {
    apply_middleware(qdrant_server::create_router(state))
}
