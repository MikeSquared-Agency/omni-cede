//! Compact page snapshot — extract structured text from the DOM.
//!
//! Inspired by Xbot's approach: instead of raw HTML, produce a compact
//! representation that an LLM can reason about efficiently.

use super::cdp::{BrowserSession, PageElement};

/// JavaScript injected into the page to extract a compact DOM snapshot.
///
/// Returns a JSON array of `{ tag, text, attributes, selector }` objects
/// for all interactive and content-bearing elements.
const SNAPSHOT_JS: &str = r#"
(() => {
    const INTERACTIVE = new Set([
        'A', 'BUTTON', 'INPUT', 'TEXTAREA', 'SELECT', 'DETAILS', 'SUMMARY'
    ]);
    const CONTENT = new Set([
        'H1','H2','H3','H4','H5','H6','P','LI','TD','TH','LABEL','SPAN',
        'STRONG','EM','CODE','PRE','BLOCKQUOTE','FIGCAPTION','ARTICLE'
    ]);
    const SKIP = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','PATH','META','LINK','BR','HR']);

    const results = [];
    const seen = new Set();
    const MAX = 300;

    function cssSelector(el) {
        if (el.id) return '#' + CSS.escape(el.id);
        let path = '';
        while (el && el !== document.body) {
            let seg = el.tagName.toLowerCase();
            if (el.id) { seg = '#' + CSS.escape(el.id); path = seg + (path ? ' > ' + path : ''); break; }
            const parent = el.parentElement;
            if (parent) {
                const siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
                if (siblings.length > 1) {
                    seg += ':nth-of-type(' + (siblings.indexOf(el) + 1) + ')';
                }
            }
            path = seg + (path ? ' > ' + path : '');
            el = parent;
        }
        return path || 'body';
    }

    function walk(node) {
        if (results.length >= MAX) return;
        if (node.nodeType !== 1) return;
        const tag = node.tagName;
        if (SKIP.has(tag)) return;
        if (node.offsetParent === null && tag !== 'BODY' && tag !== 'HTML') return; // hidden

        const isInteractive = INTERACTIVE.has(tag) || node.hasAttribute('role') ||
                              node.hasAttribute('onclick') || node.hasAttribute('tabindex');
        const isContent = CONTENT.has(tag);
        const text = (node.innerText || node.value || node.placeholder || '').trim().slice(0, 200);

        if ((isInteractive || isContent) && text.length > 0 && !seen.has(text)) {
            seen.add(text);
            const attrs = {};
            for (const a of ['href','type','name','aria-label','role','placeholder','value','alt','title','action']) {
                const v = node.getAttribute(a);
                if (v) attrs[a] = v.slice(0, 100);
            }
            results.push({
                tag: tag.toLowerCase(),
                text: text,
                attributes: attrs,
                selector: cssSelector(node),
            });
        }

        for (const child of node.children) walk(child);
    }

    walk(document.body);
    return JSON.stringify(results);
})()
"#;

/// Take a compact snapshot of the current page.
///
/// Returns a list of `PageElement` structs representing interactive and
/// content-bearing elements visible on the page.
pub async fn take_snapshot(session: &BrowserSession) -> Result<Vec<PageElement>, String> {
    let result = session.evaluate(SNAPSHOT_JS).await?;

    let json_str = result.as_str().unwrap_or("[]");
    let elements: Vec<PageElement> = serde_json::from_str(json_str).unwrap_or_default();

    Ok(elements)
}

/// Format a snapshot into a compact text representation for the LLM.
pub fn format_snapshot(elements: &[PageElement]) -> String {
    if elements.is_empty() {
        return "(empty page)".to_string();
    }

    let mut out = String::with_capacity(elements.len() * 80);
    for (i, el) in elements.iter().enumerate() {
        let attrs: Vec<String> = el
            .attributes
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        let attr_str = if attrs.is_empty() {
            String::new()
        } else {
            format!(" [{}]", attrs.join(", "))
        };

        out.push_str(&format!(
            "[{i}] <{tag}>{attr} \"{text}\" → {sel}\n",
            tag = el.tag,
            attr = attr_str,
            text = truncate(&el.text, 120),
            sel = el.selector,
        ));
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}
