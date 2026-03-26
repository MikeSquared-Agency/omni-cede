//! Anti-detection / stealth helpers.
//!
//! Provides Chrome launch flags and JavaScript patches to reduce
//! bot detection fingerprints. Inspired by Xbot's approach.

/// Chrome command-line flags that reduce automation fingerprints.
pub fn chrome_flags() -> Vec<String> {
    vec![
        "--disable-blink-features=AutomationControlled".to_string(),
        "--disable-infobars".to_string(),
        "--disable-dev-shm-usage".to_string(),
        "--disable-extensions".to_string(),
        "--disable-gpu".to_string(),
        "--no-sandbox".to_string(),
        "--disable-setuid-sandbox".to_string(),
        "--window-size=1920,1080".to_string(),
        "--start-maximized".to_string(),
    ]
}

/// JavaScript patches injected early to hide automation signals.
pub const STEALTH_JS: &str = r#"
(() => {
    // Remove webdriver flag
    Object.defineProperty(navigator, 'webdriver', { get: () => false });

    // Mock plugins (headless Chrome has none)
    Object.defineProperty(navigator, 'plugins', {
        get: () => {
            const p = { length: 3 };
            p[0] = { name: 'Chrome PDF Plugin', description: 'Portable Document Format', filename: 'internal-pdf-viewer' };
            p[1] = { name: 'Chrome PDF Viewer', description: '', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' };
            p[2] = { name: 'Native Client', description: '', filename: 'internal-nacl-plugin' };
            return p;
        }
    });

    // Mock languages
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

    // Prevent detection via permissions API
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => {
        if (parameters.name === 'notifications') {
            return Promise.resolve({ state: Notification.permission });
        }
        return originalQuery(parameters);
    };

    // Chrome runtime mock (missing in headless)
    if (!window.chrome) {
        window.chrome = {};
    }
    if (!window.chrome.runtime) {
        window.chrome.runtime = {
            connect: () => {},
            sendMessage: () => {},
        };
    }

    // Hide automation-related properties from detection scripts
    const automationProps = ['__webdriver_evaluate', '__selenium_evaluate',
        '__fxdriver_evaluate', '__driver_evaluate',
        '__webdriver_unwrapped', '__selenium_unwrapped',
        '__fxdriver_unwrapped', '__driver_unwrapped',
        '_Selenium_IDE_Recorder', '_selenium', 'calledSelenium',
        '_WEBDRIVER_ELEM_CACHE', 'ChromeDriverw',
        'driver-hierarchical', '__webdriverFunc'];
    for (const prop of automationProps) {
        delete window[prop];
        delete document[prop];
    }
})();
"#;

/// Inject stealth patches into a browser session.
///
/// Should be called immediately after page load (or via
/// `Page.addScriptToEvaluateOnNewDocument`).
pub async fn apply_stealth(session: &super::cdp::BrowserSession) -> Result<(), String> {
    // Add the script so it runs on every new document load
    session
        .send(
            "Page.addScriptToEvaluateOnNewDocument",
            serde_json::json!({ "source": STEALTH_JS }),
        )
        .await?;

    // Also inject into the current page immediately
    session.evaluate(STEALTH_JS).await?;

    Ok(())
}
