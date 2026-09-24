// Pushes the current furo colour theme into every embedded interactive-plot
// iframe. The parent can always read its own theme (unlike the iframe reading
// back across the frame boundary, which browsers block on file://), and
// postMessage to a child frame is always allowed - so this makes the plots'
// dark/light text reliable everywhere, including local file:// previews.
//
// Furo stores the theme as ``document.body.dataset.theme`` with values
// "light" | "dark" | "auto" ("auto" follows the OS prefers-color-scheme).
(function () {
    function currentTheme() {
        var t = document.body ? document.body.dataset.theme : null;
        if (t === "dark") return "dark";
        if (t === "light") return "light";
        return (window.matchMedia &&
                window.matchMedia("(prefers-color-scheme: dark)").matches)
            ? "dark" : "light";
    }
    function broadcast() {
        var theme = currentTheme();
        document.querySelectorAll("iframe").forEach(function (f) {
            try {
                f.contentWindow.postMessage({type: "monee-theme", theme: theme}, "*");
            } catch (e) {}
        });
    }
    function init() {
        // Initial pushes (cover iframes that load slightly after the page).
        broadcast();
        setTimeout(broadcast, 300);
        setTimeout(broadcast, 1000);
        // Re-push whenever an iframe finishes loading.
        document.addEventListener("load", function (e) {
            if (e.target && e.target.tagName === "IFRAME") broadcast();
        }, true);
        // Re-push on furo theme toggle (body[data-theme]) and on OS scheme change.
        if (document.body) {
            new MutationObserver(broadcast).observe(document.body,
                {attributes: true, attributeFilter: ["data-theme"]});
        }
        if (window.matchMedia) {
            try {
                window.matchMedia("(prefers-color-scheme: dark)")
                    .addEventListener("change", broadcast);
            } catch (e) {}
        }
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
