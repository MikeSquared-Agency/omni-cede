#[tokio::main]
async fn main() {
    // Initialize tracing (for tower-http TraceLayer)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "omni_cede=info,tower_http=info".parse().unwrap()),
        )
        .init();

    if let Err(e) = omni_cede::cli::run().await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}