(() => {
  "use strict";
  // Late layout (MathJax, images) moves the target after the first scroll, so
  // keep it in place until the page settles or the reader takes over.
  function pin(target) {
    const until = performance.now() + 3000;
    const inputs = ["wheel", "touchstart", "keydown", "pointerdown"];
    let done = false;
    const stop = () => {
      done = true;
      inputs.forEach((t) => removeEventListener(t, stop, true));
    };
    inputs.forEach((t) => addEventListener(t, stop, { capture: true, passive: true }));
    target.scrollIntoView({ block: "start", behavior: "instant" });
    const want = target.getBoundingClientRect().top;
    const tick = () => {
      if (done) return;
      if (performance.now() > until) return stop();
      if (Math.abs(target.getBoundingClientRect().top - want) > 1) target.scrollIntoView({ block: "start", behavior: "instant" });
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function openTarget() {
    let id;
    try {
      id = decodeURIComponent(location.hash.slice(1));
    } catch {
      return;
    }
    const target = id && document.getElementById(id);
    if (!target || !target.closest(".sd-tab-content")) return;
    let opened = false;
    for (let panel = target.closest(".sd-tab-content"); panel; panel = panel.parentElement.closest(".sd-tab-content")) {
      const label = panel.previousElementSibling;
      const input = label && label.matches(".sd-tab-label") && document.getElementById(label.htmlFor);
      if (input && !input.checked) {
        input.checked = true;
        opened = true;
      }
    }
    if (opened) pin(target);
  }

  window.addEventListener("hashchange", openTarget);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", openTarget, { once: true });
  else openTarget();
})();
