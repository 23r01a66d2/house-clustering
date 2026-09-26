/**
 * PROPERTY INTELLIGENCE — Explore Page Client Script
 * Manages multi-attribute Indian rental filtering, skeleton loaders, card rendering,
 * detail modal with Property Signature & 5 Nearest Peers, and watermark-free Leaflet mapping.
 */

let currentPage = 1;
const pageSize = 12;
let leafletMapInstance = null;
let mapMarkersGroup = null;
let mapInitialized = false;
let allMapData = null;

document.addEventListener("DOMContentLoaded", function () {
    // 1. Initial Load of Properties and Spatial Map
    fetchProperties(1);
    initLeafletMap();

    // 2. Filter Inputs Event Handlers
    const filterInputs = [
        "filterCity",
        "filterLocation",
        "filterMaxPrice",
        "filterBhk",
        "filterTypology",
        "filterFurnishing",
        "filterMinArea",
        "filterBathrooms",
        "filterSegment"
    ];

    let debounceTimer;
    filterInputs.forEach((id) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener("input", function () {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    currentPage = 1;
                    fetchProperties(1);
                }, 300);
            });
            el.addEventListener("change", function () {
                currentPage = 1;
                fetchProperties(1);
            });
        }
    });

    // 3. Reset Button
    const resetBtn = document.getElementById("resetFiltersBtn");
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            filterInputs.forEach((id) => {
                const el = document.getElementById(id);
                if (el) {
                    if (el.tagName === "SELECT") el.selectedIndex = 0;
                    else el.value = "";
                }
            });
            currentPage = 1;
            fetchProperties(1);
        });
    }

    // 4. Pagination
    const prevBtn = document.getElementById("prevPageBtn");
    const nextBtn = document.getElementById("nextPageBtn");
    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            if (currentPage > 1) fetchProperties(currentPage - 1);
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            fetchProperties(currentPage + 1);
        });
    }

    // 5. Modal Close Handlers
    const modalClose = document.getElementById("modalCloseBtn");
    const modalOverlay = document.getElementById("propertyModal");
    if (modalClose && modalOverlay) {
        modalClose.addEventListener("click", () => modalOverlay.classList.remove("active"));
        modalOverlay.addEventListener("click", (e) => {
            if (e.target === modalOverlay) modalOverlay.classList.remove("active");
        });
    }
});

// Fetch Properties via REST API
function fetchProperties(page) {
    const container = document.getElementById("propertiesContainer");
    const resultsCount = document.getElementById("resultsCount");
    const emptyState = document.getElementById("emptyState");

    // Render Skeletons during loading
    container.innerHTML = Array(6)
        .fill('<div class="skeleton-card"><div class="skeleton-shimmer"></div></div>')
        .join("");
    emptyState.style.display = "none";

    const params = new URLSearchParams({
        page: page,
        limit: pageSize,
    });

    const city = document.getElementById("filterCity")?.value;
    const location = document.getElementById("filterLocation")?.value;
    const maxPrice = document.getElementById("filterMaxPrice")?.value;
    const bhk = document.getElementById("filterBhk")?.value;
    const typology = document.getElementById("filterTypology")?.value;
    const furnishing = document.getElementById("filterFurnishing")?.value;
    const minArea = document.getElementById("filterMinArea")?.value;
    const bathrooms = document.getElementById("filterBathrooms")?.value;
    const seg = document.getElementById("filterSegment")?.value;

    if (city && city !== "All") params.append("city", city);
    if (location) params.append("location", location);
    if (maxPrice) params.append("max_price", maxPrice);
    if (bhk) params.append("bhk", bhk);
    if (typology && typology !== "All") params.append("typology", typology);
    if (furnishing && furnishing !== "All") params.append("furnishing", furnishing);
    if (minArea) params.append("min_area", minArea);
    if (bathrooms) params.append("bathrooms", bathrooms);
    if (seg !== "") params.append("segment", seg);

    fetch(`/api/properties?${params.toString()}`)
        .then((res) => res.json())
        .then((data) => {
            currentPage = data.page;

            // Update Counter
            resultsCount.innerHTML = `Showing <strong>${data.properties.length}</strong> of <strong>${data.filtered_count.toLocaleString()}</strong> rental properties (Total Inventory: ${data.total_count.toLocaleString()})`;

            // Check Empty State
            if (data.properties.length === 0) {
                container.innerHTML = "";
                emptyState.style.display = "block";
                document.getElementById("paginationControls").style.display = "none";
                return;
            }

            document.getElementById("paginationControls").style.display = "flex";

            // Render Cards with validated Indian rental attributes
            container.innerHTML = data.properties
                .map((p) => {
                    return `
                <div class="property-card">
                    <div>
                        <div class="card-top">
                            <span class="card-id">Property ID: ${p.id}</span>
                            <span class="segment-badge" style="background: rgba(30, 41, 59, 0.9); color: ${p.cluster_color}; border: 1px solid ${p.cluster_color};">
                                ● ${p.segment_name}
                            </span>
                        </div>
                        <div class="card-price">${p.price_formatted}</div>
                        <div style="font-size: 0.82rem; color: var(--emerald); font-weight: 600; margin-top: -8px; margin-bottom: 12px;">
                            ${p.location}, ${p.city}
                        </div>
                        <div class="card-specs">
                            <div class="spec-item" title="Bedrooms">
                                <i data-lucide="bed"></i>
                                <span>${p.bhk} BHK</span>
                            </div>
                            <div class="spec-item" title="Bathrooms">
                                <i data-lucide="bath"></i>
                                <span>${p.bathrooms} Baths</span>
                            </div>
                            <div class="spec-item" title="Built-Up Area">
                                <i data-lucide="ruler"></i>
                                <span>${Number(p.sqft).toLocaleString()} sqft</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.78rem; color: var(--text-dim); margin-bottom: 14px; display: flex; justify-content: space-between;">
                            <span>${p.typology}</span>
                            <span>${p.status}</span>
                        </div>
                        <button class="btn btn-outline btn-sm" style="width: 100%;" onclick="openPropertyModal('${p.id}')">
                            View Details <i data-lucide="arrow-right" style="width: 12px; height: 12px;"></i>
                        </button>
                    </div>
                </div>
            `;
                })
                .join("");

            // Update Pagination Controls
            document.getElementById("pageIndicator").innerText = `Page ${data.page} of ${data.total_pages}`;
            document.getElementById("prevPageBtn").disabled = data.page <= 1;
            document.getElementById("nextPageBtn").disabled = data.page >= data.total_pages;

            if (window.lucide) window.lucide.createIcons();
        })
        .catch((err) => {
            console.error("Error fetching properties:", err);
            container.innerHTML = `<div style="color: #EF4444; padding: 20px;">Failed to load properties: ${err.message}</div>`;
        });
}

// Open Property Modal Drawer
window.openPropertyModal = function (propId) {
    const modal = document.getElementById("propertyModal");
    const modalBody = document.getElementById("modalBody");
    modal.classList.add("active");
    modalBody.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-muted);">Fetching details for Property ID: ${propId}...</div>`;

    fetch(`/api/property/${propId}`)
        .then((res) => res.json())
        .then((p) => {
            let sigBarsHtml = p.signature.dimensions
                .map((dim) => {
                    return `
                <div class="dna-bar-row">
                    <span class="dna-bar-label">${dim.dimension}</span>
                    <div class="dna-bar-track">
                        <div class="dna-bar-fill" style="width: ${dim.score_pct}%; background: ${p.cluster_color};"></div>
                    </div>
                    <span class="dna-bar-pct mono">${dim.score_pct}%</span>
                </div>
            `;
                })
                .join("");

            modalBody.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
                <div>
                    <div style="font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; font-weight: 700;">Rental Property Record</div>
                    <h2 style="font-size: 1.8rem; font-weight: 800; color: #FFF;">Property ID: ${p.id}</h2>
                    <div style="font-size: 0.95rem; color: var(--emerald); font-weight: 600; margin-top: 2px;">
                        ${p.location}, ${p.city}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.7rem; font-weight: 800; color: var(--emerald);">${p.price_formatted}</div>
                    <span class="segment-badge" style="background: rgba(30, 41, 59, 0.9); color: ${p.cluster_color}; border: 1px solid ${p.cluster_color};">
                        ● ${p.segment_name}
                    </span>
                </div>
            </div>

            <!-- Overview Specs Grid -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; background: rgba(15, 23, 42, 0.8); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px;">
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Living Area</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">${Number(p.sqft).toLocaleString()} sqft</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Room Layout</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">${p.bhk} BHK / ${p.bathrooms} Baths</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Furnishing</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">${p.status}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Property Typology</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">${p.typology}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Rent per SqFt</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">₹${Number(p.price_per_sqft).toFixed(1)}/sqft</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Security Deposit</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">${p.deposit}</div>
                </div>
            </div>

            <!-- Normalized Property Signature -->
            <div style="margin-bottom: 28px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.05em;">
                    Normalized Property Signature
                </div>
                ${sigBarsHtml}
                <div style="font-size: 0.72rem; color: var(--text-dim); text-align: right; margin-top: 6px;">
                    *Relative position within analyzed Indian rental dataset (0-100% spectrum), NOT prediction probabilities
                </div>
            </div>

            <!-- Find Similar Properties Interactive Section -->
            <div style="border-top: 1px solid var(--border); padding-top: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #FFF; text-transform: uppercase; letter-spacing: 0.05em;">
                        Find Similar Properties
                    </div>
                    <button id="btnFindSimilarInModal" class="btn btn-outline btn-sm" onclick="fetchSimilarInModal('${p.id}')">
                        <i data-lucide="sparkles" style="width: 14px; height: 14px;"></i> Load 5 Nearest Peers
                    </button>
                </div>
                <div id="similarInModalContainer" style="color: var(--text-muted); font-size: 0.88rem;">
                    Click "Load 5 Nearest Peers" to discover the closest matching rental homes in standardized similarity space.
                </div>
            </div>
        `;
            if (window.lucide) window.lucide.createIcons();
            // Automatically fetch similar properties for immediate discovery
            fetchSimilarInModal(p.id);
        })
        .catch((err) => {
            modalBody.innerHTML = `<div style="color: #EF4444;">Failed to load details: ${err.message}</div>`;
        });
};

// Fetch Similar Properties for Modal
window.fetchSimilarInModal = function (propId) {
    const container = document.getElementById("similarInModalContainer");
    const btn = document.getElementById("btnFindSimilarInModal");
    if (btn) btn.disabled = true;
    if (container) {
        container.innerHTML = `<div style="padding: 12px; text-align: center; color: var(--text-muted);">Searching nearest neighbors in standardized similarity space...</div>`;
    }

    fetch(`/api/similar/${propId}`)
        .then((res) => res.json())
        .then((data) => {
            if (btn) btn.disabled = false;
            if (!data.similar_properties || data.similar_properties.length === 0) {
                container.innerHTML = `<div style="color: var(--text-muted); padding: 8px;">No similar properties found.</div>`;
                return;
            }

            const rowsHtml = data.similar_properties
                .map((peer) => {
                    return `
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.7); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; transition: border-color 0.2s;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <strong style="color: #FFF;">Property ID: ${peer.id}</strong>
                            <span style="background: rgba(30, 41, 59, 0.9); color: ${peer.color}; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; border: 1px solid ${peer.color};">
                                ${peer.cluster_name}
                            </span>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">
                            ${peer.price_formatted} • ${Number(peer.sqft).toLocaleString()} sqft • ${peer.bhk} BHK • ${peer.location}, ${peer.city}
                        </div>
                    </div>
                    <div style="text-align: right; display: flex; align-items: center; gap: 12px;">
                        <div>
                            <div class="mono" style="font-size: 0.95rem; font-weight: 800; color: var(--emerald);">d = ${peer.euclidean_distance}</div>
                            <div style="font-size: 0.68rem; color: var(--text-dim);">Distance</div>
                        </div>
                        <button class="btn btn-outline btn-sm" style="padding: 4px 10px; font-size: 0.75rem;" onclick="openPropertyModal('${peer.id}')">
                            Inspect
                        </button>
                    </div>
                </div>
            `;
                })
                .join("");

            container.innerHTML = `
                <div style="margin-top: 8px;">
                    <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 8px;">
                        Top 5 closest rental properties in standardized similarity space (queried Property ID: ${propId} strictly excluded):
                    </div>
                    ${rowsHtml}
                </div>
            `;
            if (window.lucide) window.lucide.createIcons();
        })
        .catch((err) => {
            if (btn) btn.disabled = false;
            container.innerHTML = `<div style="color: #EF4444; padding: 8px;">Error loading peers: ${err.message}</div>`;
        });
};

// Initialize Leaflet Map with OpenStreetMap (Zero API key dependency, Zero Watermarks)
function initLeafletMap() {
    if (mapInitialized || leafletMapInstance) return;
    const mapContainer = document.getElementById("leafletMap");
    if (!mapContainer || mapContainer._leaflet_id) return;
    mapInitialized = true;

    fetch("/api/map-properties")
        .then((res) => res.json())
        .then((data) => {
            allMapData = data;
            
            // Default center: India center covering Delhi, Mumbai, Pune
            leafletMapInstance = L.map("leafletMap", {
                scrollWheelZoom: false,
            }).setView([21.5, 75.0], 5);

            // 100% free OpenStreetMap standard tiles (Zero API key, Zero Watermarks)
            L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 18,
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
            }).addTo(leafletMapInstance);

            mapMarkersGroup = L.layerGroup().addTo(leafletMapInstance);
            renderMapMarkers(data.markers);

            setTimeout(() => {
                if (leafletMapInstance) leafletMapInstance.invalidateSize();
            }, 350);
        })
        .catch((err) => {
            console.error("Error loading map coordinates:", err);
            mapInitialized = false;
        });

    window.addEventListener("resize", () => {
        if (leafletMapInstance) leafletMapInstance.invalidateSize();
    });
}

function renderMapMarkers(markers) {
    if (!mapMarkersGroup) return;
    mapMarkersGroup.clearLayers();

    markers.forEach((m) => {
        const circle = L.circleMarker([m.lat, m.lon], {
            radius: 5,
            fillColor: m.color,
            color: "#0F172A",
            weight: 1,
            opacity: 0.9,
            fillOpacity: 0.8,
        });

        circle.bindPopup(`
            <div style="color: #0F172A; font-family: 'Plus Jakarta Sans', sans-serif;">
                <strong style="font-size: 0.95rem;">Property ID: ${m.id}</strong><br>
                <span style="font-weight: 700; color: #059669; font-size: 1rem;">${m.price_formatted}</span><br>
                <span style="font-size: 0.85rem; font-weight: 600;">${m.location}, ${m.city}</span><br>
                <span style="font-size: 0.82rem; color: #475569;">${m.bhk} BHK • ${Number(m.sqft).toLocaleString()} sqft • ${m.furnishing}</span><br>
                <span style="font-size: 0.75rem; background: #E2E8F0; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px;">
                    ${m.segment_name}
                </span>
                <div style="margin-top: 8px;">
                    <button onclick="openPropertyModal('${m.id}')" style="background: #059669; color: white; border: none; border-radius: 4px; padding: 4px 10px; font-size: 0.75rem; cursor: pointer;">
                        View Details
                    </button>
                </div>
            </div>
        `);

        mapMarkersGroup.addLayer(circle);
    });
}

// Quick Zoom to City handler
window.zoomMapToCity = function (cityName) {
    if (!leafletMapInstance || !allMapData) return;

    // Update active button state
    document.querySelectorAll(".city-map-btn").forEach((btn) => {
        if (btn.getAttribute("data-city") === cityName) btn.classList.add("active");
        else btn.classList.remove("active");
    });

    const cityBoxes = allMapData.city_boxes || {};
    if (cityName === "All") {
        leafletMapInstance.setView([21.5, 75.0], 5);
        renderMapMarkers(allMapData.markers);
    } else if (cityBoxes[cityName]) {
        const box = cityBoxes[cityName];
        leafletMapInstance.setView(box.center, box.zoom);
        // Filter markers strictly to that city
        const cityMarkers = allMapData.markers.filter((m) => m.city === cityName);
        renderMapMarkers(cityMarkers);
    }
};
