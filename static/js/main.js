/**
 * PROPERTY INTELLIGENCE — Main Client Script
 * Handles navigation, Lucide icon hydration, and subtle UI animations.
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Initialize Lucide Icons
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // 2. Mobile Navbar Toggle
    const mobileBtn = document.getElementById("mobileMenuBtn");
    const navLinks = document.getElementById("navLinks");
    if (mobileBtn && navLinks) {
        mobileBtn.addEventListener("click", function () {
            navLinks.classList.toggle("active");
        });
    }

    // 3. Navbar Background on Scroll
    const navbar = document.getElementById("mainNav");
    if (navbar) {
        window.addEventListener("scroll", function () {
            if (window.scrollY > 20) {
                navbar.style.background = "rgba(10, 15, 29, 0.95)";
                navbar.style.boxShadow = "0 4px 20px rgba(0, 0, 0, 0.4)";
            } else {
                navbar.style.background = "rgba(10, 15, 29, 0.85)";
                navbar.style.boxShadow = "none";
            }
        });
    }

    // 4. Subtle Number Count-Up for Stat Elements
    const countElements = document.querySelectorAll(".stat-value[data-count]");
    countElements.forEach((el) => {
        const target = parseFloat(el.getAttribute("data-count"));
        if (!isNaN(target) && target > 0) {
            let current = 0;
            const increment = target / 35;
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                el.innerText = Math.round(current).toLocaleString();
            }, 25);
        }
    });

    // 5. Render Math Formulas (KaTeX)
    window.renderFormulas();
    setTimeout(window.renderFormulas, 200);
    setTimeout(window.renderFormulas, 600);
});

// Helper: Render mathematical formulas cleanly via KaTeX
window.renderFormulas = function () {
    if (window.renderMathInElement) {
        window.renderMathInElement(document.body, {
            delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "\\[", right: "\\]", display: true },
                { left: "\\(", right: "\\)", display: false }
            ],
            throwOnError: false
        });
    }
};

// Helper: Re-hydrate Lucide icons dynamically added to DOM
window.refreshIcons = function () {
    if (window.lucide) {
        window.lucide.createIcons();
    }
};

// Helper: Format currency/numbers cleanly
window.formatNumber = function (val) {
    if (val === undefined || val === null || isNaN(val)) return "N/A";
    return Number(val).toLocaleString();
};
